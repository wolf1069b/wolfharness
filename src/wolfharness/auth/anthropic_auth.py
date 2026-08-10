"""Anthropic Claude Max/Pro OAuth authentication.

This module implements OAuth 2.0 PKCE authentication for Claude Max/Pro subscriptions,
allowing users to use their subscription directly through the Anthropic API.

Based on the pi-mono implementation by badlogic.
Adapted from llmling_models.auth.anthropic_auth.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import hashlib
import http.server
import json
import logging
from pathlib import Path
import secrets
import socketserver
import sys
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse
import webbrowser

import anyenv
import httpx


if TYPE_CHECKING:
    from typing import Self

from typing import Any


logger = logging.getLogger(__name__)

# OAuth client ID registered with Anthropic
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# OAuth endpoints
OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
OAUTH_MANUAL_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"

# Required scopes for API access
OAUTH_SCOPES = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)

# Beta headers required for OAuth authentication
OAUTH_BETA_HEADERS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
]

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "anthropic_oauth.json"


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (verifier, challenge)
    """
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


@dataclass
class AnthropicOAuthToken:
    """Stored OAuth token data."""

    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        """Check if the token is expired or about to expire."""
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, str | float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | float]) -> Self:
        """Create from dictionary."""
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=float(data["expires_at"]),
        )


@dataclass
class AnthropicTokenStore:
    """File-based token storage for Anthropic OAuth."""

    path: Path = field(default_factory=lambda: DEFAULT_TOKEN_PATH)
    _token: AnthropicOAuthToken | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Ensure storage directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AnthropicOAuthToken | None:
        """Load token from file."""
        if self._token is not None:
            return self._token

        if not self.path.exists():
            return None

        try:
            data = anyenv.load_json(self.path.read_text(), return_type=dict)
            self._token = AnthropicOAuthToken.from_dict(data)
        except (anyenv.JsonLoadError, KeyError, TypeError) as e:
            logger.warning("Failed to load token from %s: %s", self.path, e)
            return None
        else:
            return self._token

    def save(self, token: AnthropicOAuthToken) -> None:
        """Save token to file."""
        self._token = token
        self.path.write_text(json.dumps(token.to_dict(), indent=2))
        self.path.chmod(0o600)
        logger.debug("Saved token to %s", self.path)

    def clear(self) -> None:
        """Remove stored token."""
        self._token = None
        if self.path.exists():
            self.path.unlink()
            logger.debug("Removed token from %s", self.path)

    def get_valid_token(self) -> AnthropicOAuthToken | None:
        """Get token if it exists and is not expired."""
        token = self.load()
        if token is None:
            return None
        if token.is_expired():
            logger.debug("Token is expired, needs refresh")
            return None
        return token


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    code: str | None = None
    state: str | None = None
    error: str | None = None
    done_event: threading.Event = threading.Event()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress default logging."""

    def do_GET(self) -> None:
        """Handle GET request for OAuth callback."""
        parsed = urlparse(self.path)

        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)

        if "error" in params:
            _OAuthCallbackHandler.error = params["error"][0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<h1>Authentication Failed</h1><p>Error: {params['error'][0]}</p>"
                "<p>You can close this window.</p>".encode()
            )
            _OAuthCallbackHandler.done_event.set()
            return

        if "code" in params and "state" in params:
            _OAuthCallbackHandler.code = params["code"][0]
            _OAuthCallbackHandler.state = params["state"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authentication Successful</h1>"
                b"<p>You can close this window and return to the terminal.</p>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authentication Failed</h1><p>Missing code or state parameter.</p>"
            )
        _OAuthCallbackHandler.done_event.set()


def _start_callback_server() -> tuple[socketserver.TCPServer, threading.Thread, int]:
    """Start local HTTP server for OAuth callback on a dynamic port.

    Returns:
        Tuple of (server, thread, port)
    """
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None
    _OAuthCallbackHandler.error = None
    _OAuthCallbackHandler.done_event.clear()

    server = socketserver.TCPServer(("127.0.0.1", 0), _OAuthCallbackHandler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    return server, thread, port


def build_authorization_url(verifier: str, challenge: str, redirect_uri: str) -> str:
    """Build the OAuth authorization URL."""
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{OAUTH_AUTHORIZE_URL}?{query}"


def exchange_code_for_token(
    code: str, state: str, verifier: str, redirect_uri: str
) -> AnthropicOAuthToken:
    """Exchange authorization code for access token.

    Args:
        code: The authorization code from callback
        state: The OAuth state from callback
        verifier: The PKCE code verifier
        redirect_uri: The redirect URI used in the authorization request

    Returns:
        The OAuth token
    """
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
        )

        if not response.is_success:
            msg = f"Token exchange failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        expires_at = time.time() + data["expires_in"] - 300

        return AnthropicOAuthToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=expires_at,
        )


def refresh_access_token(refresh_token: str) -> AnthropicOAuthToken:
    """Refresh an expired access token."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
        )

        if not response.is_success:
            msg = f"Token refresh failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        expires_at = time.time() + data["expires_in"] - 300

        return AnthropicOAuthToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=expires_at,
        )


def authenticate_anthropic_max(  # noqa: PLR0915
    verbose: bool = True,
    open_browser: bool = True,
) -> AnthropicOAuthToken:
    """Authenticate with Anthropic using OAuth for Claude Max/Pro.

    Uses a local callback server to automatically capture the authorization code.
    Falls back to manual paste if the callback fails.
    """
    verifier, challenge = generate_pkce()

    if verbose:
        print("Starting local server for OAuth callback...")
    server, _thread, port = _start_callback_server()
    redirect_uri = f"http://localhost:{port}/callback"

    try:
        auth_url = build_authorization_url(verifier, challenge, redirect_uri)

        if verbose:
            print("\nTo authenticate with Claude Max/Pro:")
            print(f"\n1. Visit: {auth_url}")
            print("\n2. Sign in with your Anthropic account")
            print("3. The callback will be captured automatically")
            print()

        if open_browser:
            if verbose:
                print("Opening browser...")
            webbrowser.open(auth_url)

        if verbose:
            print("Waiting for OAuth callback...")
        _OAuthCallbackHandler.done_event.wait(timeout=300)

        if _OAuthCallbackHandler.error:
            msg = f"OAuth error: {_OAuthCallbackHandler.error}"
            raise RuntimeError(msg)

        code = _OAuthCallbackHandler.code
        state = _OAuthCallbackHandler.state

        # Fall back to manual paste if callback wasn't received
        if not code or not state:
            print("\nCallback not received. Paste the authorization code or full redirect URL:")
            try:
                user_input = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nAuthentication cancelled.")
                msg = "Authentication cancelled by user"
                raise RuntimeError(msg) from None

            if not user_input:
                msg = "No authorization code provided"
                raise RuntimeError(msg)

            # Parse input - could be code, code#state, or full URL
            parsed = _parse_authorization_input(user_input)
            code = parsed.get("code")
            state = parsed.get("state", verifier)

            if not code:
                msg = "Could not extract authorization code from input"
                raise RuntimeError(msg)

        # Verify state
        if state != verifier:
            msg = "OAuth state mismatch - possible CSRF attack"
            raise RuntimeError(msg)

        if verbose:
            print("\nExchanging code for token...")

        token = exchange_code_for_token(code, state, verifier, redirect_uri)

        if verbose:
            print("Authentication successful!")

        return token

    finally:
        server.shutdown()
        server.server_close()


def _parse_authorization_input(user_input: str) -> dict[str, str]:
    """Parse authorization input which may be a code, code#state, or full URL."""
    value = user_input.strip()
    if not value:
        return {}

    # Try as URL
    try:
        parsed = urlparse(value)
        params = parse_qs(parsed.query)
        result: dict[str, str] = {}
        if "code" in params:
            result["code"] = params["code"][0]
        if "state" in params:
            result["state"] = params["state"][0]
        if result:
            return result
    except Exception:  # noqa: BLE001
        pass

    # Try code#state format
    if "#" in value:
        parts = value.split("#", maxsplit=1)
        return {"code": parts[0], "state": parts[1]}

    # Try query string format
    if "code=" in value:
        params = parse_qs(value)
        result = {}
        if "code" in params:
            result["code"] = params["code"][0]
        if "state" in params:
            result["state"] = params["state"][0]
        if result:
            return result

    # Plain code
    return {"code": value}


def get_or_refresh_token(
    store: AnthropicTokenStore | None = None,
) -> AnthropicOAuthToken:
    """Get a valid token, refreshing if necessary."""
    if store is None:
        store = AnthropicTokenStore()

    token = store.load()
    if token is None:
        msg = "No Anthropic OAuth token found. Run anthropic-auth to authenticate."
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        token = refresh_access_token(token.refresh_token)
        store.save(token)

    return token


async def get_or_refresh_token_async(
    store: AnthropicTokenStore | None = None,
) -> AnthropicOAuthToken:
    """Async version of get_or_refresh_token."""
    if store is None:
        store = AnthropicTokenStore()

    token = store.load()
    if token is None:
        msg = "No Anthropic OAuth token found. Run anthropic-auth to authenticate."
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OAUTH_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": token.refresh_token,
                    "client_id": CLIENT_ID,
                },
            )

            if not response.is_success:
                msg = f"Token refresh failed: {response.status_code} - {response.text}"
                raise RuntimeError(msg)

            data = response.json()
            expires_at = time.time() + data["expires_in"] - 300

            token = AnthropicOAuthToken(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=expires_at,
            )
            store.save(token)

    return token


def anthropic_auth_main() -> None:
    """Command-line entry point for Anthropic OAuth authentication."""
    parser = argparse.ArgumentParser(
        description="Authenticate with Anthropic Claude Max/Pro using OAuth."
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser",
    )
    parser.add_argument(
        "--token-path",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help=f"Path to store token (default: {DEFAULT_TOKEN_PATH})",
    )
    parser.add_argument(
        "--logout",
        action="store_true",
        help="Remove stored token and log out",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current authentication status",
    )

    args = parser.parse_args()
    store = AnthropicTokenStore(path=args.token_path)

    if args.logout:
        store.clear()
        print("Logged out. Token removed.")
        return

    if args.status:
        token = store.load()
        if token is None:
            print("Not authenticated.")
            print(f"Token path: {args.token_path}")
            sys.exit(1)
        elif token.is_expired():
            print("Token expired. Run without --status to refresh.")
            sys.exit(1)
        else:
            remaining = token.expires_at - time.time()
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            print(f"Authenticated. Token expires in {hours}h {minutes}m.")
            print(f"Token path: {args.token_path}")
        return

    try:
        token = authenticate_anthropic_max(
            verbose=True,
            open_browser=not args.no_browser,
        )
        store.save(token)
        print(f"\nToken saved to: {args.token_path}")
        print("You can now use Claude Max/Pro models with auth_method='oauth'")
    except Exception as e:
        logger.exception("Authentication failed")
        print(f"\nAuthentication failed: {e}", file=sys.stderr)
        sys.exit(1)


# --- AnthropicMaxProvider ---

# Required system prompt prefix for OAuth validation
# Anthropic checks for this to validate the token is being used by "Claude Code"
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# Version string to match Claude Code CLI
CC_VERSION = "2.1.87"

# CCH (checksum hash) constants
_CCH_SEED_B64 = b"blJzasgGgx4="
CCH_MASK = 0xFFFFF
CCH_PLACEHOLDER = "cch=00000"

# Fingerprint salt for billing header
FINGERPRINT_SALT = "59cf53e54c78"


def _compute_fingerprint(first_user_message: str) -> str:
    """Compute fingerprint from first user message for billing header.

    Takes chars at indices 4, 7, 20 from the message, combines with
    a salt and version string, then SHA-256 hashes it.

    Args:
        first_user_message: The first user message content

    Returns:
        3-char hex fingerprint string
    """
    indices = [4, 7, 20]
    chars = "".join(first_user_message[i] if i < len(first_user_message) else "0" for i in indices)
    input_str = f"{FINGERPRINT_SALT}{chars}{CC_VERSION}"
    return hashlib.sha256(input_str.encode()).hexdigest()[:3]


def _compute_cch(body: bytes) -> str:
    """Compute CCH checksum over request body using xxhash64.

    Args:
        body: Request body bytes

    Returns:
        5-char hex checksum string

    Raises:
        ImportError: If xxhash package is not installed
    """
    try:
        import xxhash  # type: ignore[import-not-found]
    except ImportError:
        msg = "xxhash package required for Anthropic Max OAuth. Install with: pip install xxhash"
        raise ImportError(msg) from None

    seed = int.from_bytes(base64.b64decode(_CCH_SEED_B64), "big")
    hash_value = xxhash.xxh64(body, seed=seed).intdigest()
    return f"{hash_value & CCH_MASK:05x}"


class AnthropicMaxHTTPClient(httpx.AsyncClient):
    """Custom HTTP client that injects OAuth Bearer token and beta headers.

    This client:
    - Adds Authorization: Bearer <access_token> header
    - Adds required anthropic-beta headers for OAuth
    - Injects "You are Claude Code" system prompt (required for OAuth validation)
    - Adds ?beta=true query parameter to match Claude Code
    - Sets user-agent to identify as Claude Code CLI
    - Automatically refreshes expired tokens
    """

    def __init__(
        self,
        token_store: AnthropicTokenStore,
        **kwargs: Any,
    ) -> None:
        """Initialize the client.

        Args:
            token_store: Token store for retrieving/refreshing tokens
            **kwargs: Additional arguments passed to AsyncClient
        """
        super().__init__(**kwargs)
        self.token_store = token_store
        self._cached_token: AnthropicOAuthToken | None = None

    async def _get_token(self) -> AnthropicOAuthToken:
        """Get a valid token, using cache when possible."""
        # Check if cached token is still valid
        if self._cached_token is not None and not self._cached_token.is_expired():
            return self._cached_token

        # Get or refresh token
        self._cached_token = await get_or_refresh_token_async(self.token_store)
        return self._cached_token

    def _inject_claude_code_system(self, body: bytes) -> bytes:
        """Inject Claude Code system prompt and billing header.

        Anthropic's OAuth validation requires:
        1. System prompt containing "You are Claude Code" as a SEPARATE text block
        2. A billing header block with version, fingerprint, and CCH placeholder

        The CCH placeholder is later replaced with the actual checksum.

        Args:
            body: Original request body

        Returns:
            Modified request body with Claude Code system prompt and billing header
        """
        import json

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return body

        # Only modify messages API requests
        if "messages" not in data:
            return body

        system = data.get("system", "")
        needs_claude_code = "Claude Code" not in str(system)

        # Build system blocks list
        blocks: list[dict[str, Any]] = []

        # 1. Billing header block (with CCH placeholder to be replaced later)
        first_user_msg = self._extract_first_user_text(data.get("messages", []))
        fingerprint = _compute_fingerprint(first_user_msg)
        billing_text = (
            f"x-anthropic-billing-header: cc_version={CC_VERSION}.{fingerprint}; "
            f"cc_entrypoint=cli; {CCH_PLACEHOLDER};"
        )
        blocks.append({"type": "text", "text": billing_text})

        # 2. Claude Code system prompt
        if needs_claude_code:
            blocks.append({"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX})

        # 3. Existing system content
        if isinstance(system, str) and system:
            blocks.append({"type": "text", "text": system})
        elif isinstance(system, list):
            blocks.extend(system)

        data["system"] = blocks

        logger.debug("Injected Claude Code system prompt and billing header")
        return json.dumps(data).encode()

    @staticmethod
    def _extract_first_user_text(messages: list[Any]) -> str:
        """Extract text from the first user message."""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return str(block.get("text", ""))
        return ""

    @staticmethod
    def _apply_cch(body: bytes) -> bytes:
        """Compute CCH checksum and replace placeholder in body.

        Args:
            body: Request body containing CCH_PLACEHOLDER

        Returns:
            Body with placeholder replaced by actual checksum
        """
        body_str = body.decode()
        if CCH_PLACEHOLDER not in body_str:
            return body
        cch = _compute_cch(body)
        return body_str.replace(CCH_PLACEHOLDER, f"cch={cch}").encode()

    async def send(self, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        """Send request with OAuth headers and system prompt injected.

        Args:
            request: The HTTP request to send
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            The HTTP response
        """
        import httpx

        token = await self._get_token()

        # Set Authorization header (Bearer token, not API key)
        request.headers["authorization"] = f"Bearer {token.access_token}"

        # Remove x-api-key if present (SDK might add it)
        if "x-api-key" in request.headers:
            del request.headers["x-api-key"]

        # Set user-agent to identify as Claude Code CLI (required for OAuth validation)
        # Anthropic checks this to ensure the token is being used by Claude Code
        request.headers["user-agent"] = f"claude-cli/{CC_VERSION} (external, cli)"
        request.headers["x-app"] = "cli"

        # Add ?beta=true query parameter to match Claude Code endpoint
        # This is critical - without it, Anthropic rejects OAuth tokens
        url = str(request.url)
        if "?" not in url:
            url = f"{url}?beta=true"
        elif "beta=true" not in url:
            url = f"{url}&beta=true"

        # Merge beta headers with any existing ones
        existing_beta = request.headers.get("anthropic-beta", "")
        existing_list = [b.strip() for b in existing_beta.split(",") if b.strip()]

        # Combine and deduplicate
        all_betas = list(dict.fromkeys(OAUTH_BETA_HEADERS + existing_list))
        request.headers["anthropic-beta"] = ",".join(all_betas)

        # Inject Claude Code system prompt and billing header into request body,
        # then compute CCH checksum and replace placeholder.
        if request.content:
            modified_body = self._inject_claude_code_system(request.content)
            # Compute CCH over the body (with placeholder) and replace it
            modified_body = self._apply_cch(modified_body)
            # Rebuild request with modified URL, body and updated headers
            new_request = httpx.Request(
                method=request.method,
                url=url,
                headers=dict(request.headers),
                content=modified_body,
            )
            new_request.headers["content-length"] = str(len(modified_body))
            logger.debug(
                "Sending request with OAuth authentication and Claude Code spoof to %s",
                url,
            )
            return await super().send(new_request, *args, **kwargs)

        # Rebuild request with modified URL even if no body
        new_request = httpx.Request(
            method=request.method,
            url=url,
            headers=dict(request.headers),
        )
        logger.debug("Sending request with OAuth authentication to %s", url)
        return await super().send(new_request, *args, **kwargs)


def _create_client(token_store: AnthropicTokenStore) -> Any:
    """Create Anthropic client with OAuth-enabled HTTP client.

    Args:
        token_store: Token store for authentication

    Returns:
        Configured AsyncAnthropic client
    """
    try:
        from anthropic import AsyncAnthropic

        http_client = AnthropicMaxHTTPClient(token_store, timeout=600.0)
        return AsyncAnthropic(
            api_key="oauth-placeholder",  # Required by SDK but not used
            http_client=http_client,
        )
    except ImportError:
        msg = "anthropic package required. Install with: pip install anthropic"
        raise ImportError(msg) from None


class AnthropicMaxProvider:
    """Provider for Anthropic API using Claude Max/Pro OAuth authentication.

    This provider allows Claude Max/Pro subscribers to use their subscription
    through the Anthropic API instead of requiring a separate API key.
    """

    def __init__(self, token_store: AnthropicTokenStore | None = None) -> None:
        """Initialize the provider.

        Args:
            token_store: Custom token store (defaults to standard location)
        """
        self._token_store = token_store or AnthropicTokenStore()
        self._client: Any = None

    @property
    def name(self) -> str:
        """The provider name."""
        return "anthropic-max"

    @property
    def base_url(self) -> str:
        """The base URL for the Anthropic API."""
        return "https://api.anthropic.com"

    @property
    def client(self) -> Any:
        """Get the Anthropic client with OAuth authentication."""
        if self._client is None:
            self._client = _create_client(self._token_store)
        return self._client


if __name__ == "__main__":
    anthropic_auth_main()
