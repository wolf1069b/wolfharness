"""Models for agent configuration."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, Literal, assert_never
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator
from pydantic_ai import UsageLimits  # noqa: TC002
from schemez import InlineSchemaDef
from tokonomics.model_discovery import ProviderType  # noqa: TC002
from tokonomics.model_names import ModelId  # noqa: TC002
from toprompt import render_prompt

from wolfharness import log
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.common_types import EndStrategy  # noqa: TC001
from wolfharness.models.fields import OutputTypeField, SystemPromptField  # noqa: TC001
from wolfharness.models.model_configs import AnyModelConfig  # noqa: TC001
from wolfharness.prompts.prompts import PromptMessage, StaticPrompt
from wolfharness_config import BaseToolConfig, NativeAgentToolConfig
from wolfharness_config.builtin_tools import BaseBuiltinToolConfig
from wolfharness_config.knowledge import Knowledge  # noqa: TC001
from wolfharness_config.nodes import BaseAgentConfig
from wolfharness_config.session import MemoryConfig, SessionQuery
from wolfharness_config.toolsets import BaseToolsetConfig, ToolsetConfig
from wolfharness_config.workers import WorkerConfig  # noqa: TC001


if TYPE_CHECKING:
    import builtins

    from pydantic_ai.capabilities import AbstractCapability

    from wolfharness.agents.native_agent import Agent
    from wolfharness.common_types import AnyEventHandlerType
    from wolfharness.delegation import AgentPool
    from wolfharness.prompts.prompts import BasePrompt
    from wolfharness.tools.base import Tool
    from wolfharness.ui.base import InputProvider
    from wolfharness_config.workers import AgentWorkerConfig

ToolMode = Literal["codemode"]

logger = log.get_logger(__name__)

# Unified type for all tool configurations (single tools + toolsets)
AnyToolConfig = Annotated[NativeAgentToolConfig | ToolsetConfig, Field(discriminator="type")]


class NativeAgentConfig(BaseAgentConfig):
    """Configuration for a single agent in the system.

    Defines an agent's complete configuration including its model, environment,
    and behavior settings.

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/agent_configuration/
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:hubot-16",
            "x-doc-title": "Agent Configuration",
        }
    )

    type: Literal["native"] = Field(default="native", init=False)
    """Top-level discriminator for agent type."""

    model: AnyModelConfig | ModelId | str = Field(
        ...,
        examples=["openai:gpt-5-nano"],
        title="Model configuration or name",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/wolfharness/YAML%20Configuration/model_configuration/"
        },
    )
    """The model to use for this agent. Can be either a simple model name
    string (e.g. 'openai:gpt-5') or a structured model definition.

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/model_configuration/
    """

    tools: Sequence[AnyToolConfig | str] = Field(
        default_factory=list,
        examples=[
            ["webbrowser:open", "builtins:print"],
            [
                {
                    "type": "import",
                    "import_path": "webbrowser:open",
                    "name": "web_browser",
                },
                {
                    "type": "bash",
                    "timeout": 30.0,
                },
            ],
        ],
        title="Tool configurations",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/wolfharness/YAML%20Configuration/tool_configuration/"
        },
    )
    """A list of tools and toolsets to register with this agent.

    Supports both single tools (bash, import, web_search, etc.) and
    toolsets (file_access, process_management, code, etc.).

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/tool_configuration/
    """

    session: str | SessionQuery | MemoryConfig | None = Field(
        default=None,
        examples=["main_session", "user_123"],
        title="Session configuration",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/wolfharness/YAML%20Configuration/session_configuration/"
        },
    )
    """Session configuration for conversation recovery.

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/session_configuration/
    """

    output_type: OutputTypeField = None

    retries: int = Field(default=1, ge=0, examples=[1, 3], title="Model retries")
    """Number of retries for failed operations (maps to pydantic-ai's retries)"""

    output_retries: int | None = Field(default=None, examples=[1, 3], title="Output retries")
    """Max retries for result validation"""

    end_strategy: EndStrategy = Field(
        default="early",
        examples=["early", "exhaust"],
        title="Tool execution strategy",
    )
    """The strategy for handling multiple tool calls when a final result is found"""

    avatar: str | None = Field(
        default=None,
        examples=["https://example.com/avatar.png", "/assets/robot.jpg"],
        title="Avatar image",
    )
    """URL or path to agent's avatar image"""

    system_prompt: SystemPromptField = None
    """System prompt for the agent. Can be a string or list of strings/prompt configs.

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/system_prompts_configuration/
    """

    # context_sources: list[ContextSource] = Field(default_factory=list)
    # """Initial context sources to load"""

    knowledge: Knowledge | None = Field(
        default=None,
        title="Knowledge sources",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/wolfharness/YAML%20Configuration/knowledge_configuration/"
        },
    )
    """Knowledge sources for this agent.

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/knowledge_configuration/
    """

    workers: list[WorkerConfig] = Field(
        default_factory=list,
        examples=[
            [{"type": "agent", "name": "web_agent", "reset_history_on_run": True}],
            [{"type": "team", "name": "analysis_team"}],
        ],
        title="Worker agents",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/wolfharness/YAML%20Configuration/worker_configuration/"
        },
    )
    """Worker agents which will be available as tools.

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/worker_configuration/
    """

    usage_limits: UsageLimits | None = Field(default=None, title="Usage limits")
    """Usage limits for this agent."""

    model_providers: list[ProviderType] | None = Field(
        default=None,
        examples=[["models.dev"], ["anthropic", "openai"]],
        title="Model providers",
    )
    """List of model providers to use for model discovery.

    When set, the agent's get_available_models() will return models from these
    providers. Common values: "openai", "anthropic", "gemini", "mistral", etc.
    If not set, defaults to ["models.dev"].
    """

    tool_mode: ToolMode | None = Field(
        default=None,
        examples=["codemode"],
        title="Tool execution mode",
    )
    """Tool execution mode:
    - None: Default mode - tools are called directly
    - "codemode": Tools are wrapped in a Python execution environment
    """

    capabilities: list[Any] = Field(default_factory=list)
    """Additional pydantic-ai capabilities to attach to the agent.

    Can contain either CapabilityConfig objects (for YAML-loaded capabilities)
    or pre-instantiated AbstractCapability objects (for Python API usage).
    """

    @model_validator(mode="before")
    @classmethod
    def validate_output_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert result type and apply its settings."""
        output_type = data.get("output_type")
        if isinstance(output_type, dict):
            # Extract response-specific settings
            tool_name = output_type.pop("result_tool_name", None)
            tool_description = output_type.pop("result_tool_description", None)
            retries = output_type.pop("output_retries", None)

            # Convert remaining dict to ResponseDefinition
            if "type" not in output_type["response_schema"]:
                output_type["response_schema"]["type"] = "inline"
            data["output_type"]["response_schema"] = InlineSchemaDef(**output_type)

            # Apply extracted settings to agent config
            if tool_name:
                data["result_tool_name"] = tool_name
            if tool_description:
                data["result_tool_description"] = tool_description
            if retries is not None:
                data["output_retries"] = retries

        return data

    @model_validator(mode="before")
    @classmethod
    def handle_model_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert model inputs to appropriate format."""
        if isinstance((model := data.get("model")), str):
            data["model"] = {"type": "string", "identifier": model}
        return data

    @model_validator(mode="before")
    @classmethod
    def handle_capabilities(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert capability dicts to appropriate config models.

        Resolution order for each capability dict's ``type`` field:

        1. **Built-in short name** (e.g. ``"loop_detection"``) → validated
           against the typed built-in config model.
        2. **Entry-point name** (e.g. ``"mermaid_lint"``) → wrapped in
           :class:`EntryPointCapabilityConfig` for later resolution via the
           ``wolfharness.capabilities`` entry-point group.
        3. **Python import path** (e.g.
           ``"pydantic_ai.capabilities.Instrumentation"``) → wrapped in
           :class:`GenericCapabilityConfig` for ``__import__`` resolution.
        """
        from pydantic import TypeAdapter

        from wolfharness_config.capabilities import (
            CapabilityConfig,
            EntryPointCapabilityConfig,
            GenericCapabilityConfig,
            is_known_capability_type,
        )

        if capabilities := data.get("capabilities"):
            # Discover entry-point capabilities once for this validation pass.
            ep_names: set[str] | None = None
            resolved: list[Any] = []
            _adapter: TypeAdapter[CapabilityConfig] = TypeAdapter(CapabilityConfig)
            for cap in capabilities:
                if isinstance(cap, dict):
                    raw_type = cap.get("type", "")
                    if is_known_capability_type(raw_type):
                        resolved.append(_adapter.validate_python(cap))
                    else:
                        # Lazily discover entry-point names only when needed.
                        if ep_names is None:
                            try:
                                from wolfharness.capabilities.registry import (
                                    discover_entry_point_capabilities,
                                )

                                ep_names = set(discover_entry_point_capabilities().keys())
                            except Exception:  # noqa: BLE001
                                ep_names = set()

                        if raw_type in ep_names:
                            resolved.append(EntryPointCapabilityConfig(**cap))
                        else:
                            resolved.append(GenericCapabilityConfig(**cap))
                else:
                    resolved.append(cap)
            data["capabilities"] = resolved
        return data

    @model_validator(mode="before")
    @classmethod
    def handle_workers_shorthand(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert worker string shorthand to AgentWorkerConfig objects.

        Supports:
        - Plain string: "worker_name" -> {"type": "agent", "name": "worker_name"}
        - Dict with type: {"type": "agent", "name": "worker_name"} (unchanged)
        """
        if workers := data.get("workers"):
            resolved_workers: list[dict[str, Any] | AgentWorkerConfig] = []
            for worker in workers:
                if isinstance(worker, str):
                    # Convert plain string to AgentWorkerConfig dict
                    resolved_workers.append({"type": "agent", "name": worker})
                else:
                    resolved_workers.append(worker)
            data["workers"] = resolved_workers
        return data

    def get_agent[TDeps](
        self,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        pool: AgentPool[Any] | None = None,
        deps_type: builtins.type[TDeps] | None = None,
    ) -> Agent[TDeps, Any]:
        from wolfharness.agents.native_agent import Agent

        return Agent[TDeps].from_config(
            self,
            event_handlers=event_handlers,
            input_provider=input_provider,
            agent_pool=pool,
            deps_type=deps_type,
        )

    def get_tool_providers(self) -> list[AbstractCapability]:
        """Get all resource providers for this agent's tools.

        Processes the unified tools list, separating:
        - Toolsets: Each becomes its own AbstractCapability
        - Single tools: Aggregated into a single FunctionToolsetCapability

        Returns:
            List of AbstractCapability instances
        """
        from wolfharness.tools.base import Tool

        providers: list[AbstractCapability] = []
        static_tools: list[Tool] = []

        for tool_config in self.tools:
            # Skip builtin tools - they're handled via get_builtin_tools()
            if isinstance(tool_config, BaseBuiltinToolConfig):
                continue
            if isinstance(tool_config, BaseToolsetConfig):
                # Toolset -> get its provider directly
                providers.append(tool_config.get_provider())
            elif isinstance(tool_config, str):
                # String import path -> single tool
                static_tools.append(Tool.from_callable(tool_config))
            elif isinstance(tool_config, BaseToolConfig):
                # Single tool config -> single tool
                static_tools.append(tool_config.get_tool())

        # Wrap all single tools in one provider
        if static_tools:
            providers.append(FunctionToolsetCapability(name="tools", tools=static_tools))

        return providers

    # Keep old methods for backward compatibility during transition
    def get_toolsets(self) -> list[AbstractCapability]:
        """Get toolset providers. Deprecated: use get_tool_providers() instead."""
        return [
            p
            for p in self.get_tool_providers()
            if not isinstance(p, FunctionToolsetCapability) or p.name != "tools"
        ]

    def get_tool_provider(self) -> AbstractCapability | None:
        """Get single tools provider. Deprecated: use get_tool_providers() instead."""
        for p in self.get_tool_providers():
            if isinstance(p, FunctionToolsetCapability) and p.name == "tools":
                return p
        return None

    def get_builtin_tools(self) -> list[Any]:
        """Get pydantic-ai builtin tools from config.

        Returns:
            List of AbstractBuiltinTool instances (WebSearchTool, etc.)
        """
        builtin_tools: list[Any] = []
        for tool_config in self.tools:
            if isinstance(tool_config, BaseBuiltinToolConfig):
                try:
                    builtin_tools.append(tool_config.get_builtin_tool())
                except Exception:
                    logger.exception("Failed to load builtin tool", config=tool_config)
        return builtin_tools

    def get_session_config(self) -> MemoryConfig:
        """Get resolved memory configuration."""
        match self.session:
            case str() | UUID():
                return MemoryConfig(session=SessionQuery(name=str(self.session)))
            case SessionQuery():
                return MemoryConfig(session=self.session)
            case MemoryConfig():
                return self.session
            case None:
                return MemoryConfig()
            case _ as unreachable:
                assert_never(unreachable)

    def get_history_processors(self) -> list[Callable[..., Any]]:
        """Get resolved history processors from session config.

        Returns:
            List of processor callables

        Raises:
            ValueError: If processor resolution fails or signature is invalid
        """
        import inspect

        from wolfharness.utils.importing import import_callable

        # Get session config
        memory_cfg = self.get_session_config()

        # Get processor paths from config
        processor_paths = getattr(memory_cfg, "history_processors", None)
        if not processor_paths:
            return []

        # Resolve import paths to callables
        resolved: list[Callable[..., Any]] = []
        for path in processor_paths:
            try:
                processor = import_callable(path)

                # Validate signature
                sig = inspect.signature(processor)
                params = list(sig.parameters.values())
            except Exception as e:
                msg = f"Failed to resolve history processor '{path}': {e}"
                raise ValueError(msg) from e

            # Check parameter count (outside try to avoid catching our own raise)
            if len(params) not in (1, 2):
                msg = f"History processor must take 1 or 2 arguments, got {len(params)}"
                raise ValueError(msg)

            resolved.append(processor)

        return resolved

    def get_system_prompts(self) -> list[BasePrompt]:
        """Get all system prompts as BasePrompts."""
        from wolfharness_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            PackagePromptConfig,
            StaticPromptConfig,
        )

        prompts: list[BasePrompt] = []
        # Normalize system_prompt to a list
        if self.system_prompt is None:
            return prompts
        prompt_list = (
            [self.system_prompt] if isinstance(self.system_prompt, str) else self.system_prompt
        )
        for prompt in prompt_list:
            match prompt:
                case (str() as content) | StaticPromptConfig(content=content):
                    # Convert string to StaticPrompt
                    msgs = [PromptMessage(role="system", content=content)]
                    static = StaticPrompt(name="system", description="System prompt", messages=msgs)
                    prompts.append(static)
                case FilePromptConfig(path=path):
                    template_content = path.read_text("utf-8")
                    # Create a template-based prompt (for now as StaticPrompt with placeholder)
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"File prompt: {path}",
                        messages=[PromptMessage(role="system", content=template_content)],
                    )
                    prompts.append(static_prompt)
                case LibraryPromptConfig(reference=ref):
                    # Create placeholder for library prompts (resolved by manifest)
                    msg = PromptMessage(role="system", content=f"[LIBRARY:{ref}]")
                    static = StaticPrompt(name="system", description=f"Ref: {ref}", messages=[msg])
                    prompts.append(static)
                case FunctionPromptConfig(arguments=arguments, function=function):
                    # Import and call the function to get prompt content
                    content = function(**arguments)
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"Function prompt: {function}",
                        messages=[PromptMessage(role="system", content=content)],
                    )
                    prompts.append(static_prompt)
                case PackagePromptConfig(package=pkg, resource=resource):
                    from importlib.resources import files as pkg_files

                    template_content = (pkg_files(pkg) / resource).read_text(encoding="utf-8")
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"Package prompt: {pkg}/{resource}",
                        messages=[PromptMessage(role="system", content=template_content)],
                    )
                    prompts.append(static_prompt)
                case _ as unreachable:
                    assert_never(unreachable)  # ty: ignore[type-assertion-failure]
        return prompts

    def render_system_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        """Render system prompts with context."""
        from wolfharness_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            PackagePromptConfig,
            StaticPromptConfig,
        )

        context = context or {"name": self.name, "id": 1, "model": self.model}
        rendered_prompts: list[str] = []
        # Normalize system_prompt to a list
        if self.system_prompt is None:
            return rendered_prompts
        prompt_list = (
            [self.system_prompt] if isinstance(self.system_prompt, str) else self.system_prompt
        )
        for prompt in prompt_list:
            match prompt:
                case (str() as content) | StaticPromptConfig(content=content):
                    rendered_prompts.append(render_prompt(content, {"agent": context}))
                case FilePromptConfig(path=path, variables=variables):
                    # Load and render Jinja template from file
                    template_content = path.read_text("utf-8")
                    template_ctx = {"agent": context, **variables}
                    rendered_prompts.append(render_prompt(template_content, template_ctx))
                case LibraryPromptConfig(reference=reference):
                    # This will be handled by the manifest's get_agent method
                    # For now, just add a placeholder
                    rendered_prompts.append(f"[LIBRARY:{reference}]")
                case FunctionPromptConfig(function=function, arguments=arguments):
                    # Import and call the function to get prompt content
                    content = function(**arguments)
                    rendered_prompts.append(render_prompt(content, {"agent": context}))
                case PackagePromptConfig(
                    package=pkg,
                    resource=resource,
                    variables=variables,
                ):
                    from importlib.resources import files as pkg_files

                    template_content = (pkg_files(pkg) / resource).read_text(encoding="utf-8")
                    template_ctx = {"agent": context, **variables}
                    rendered_prompts.append(render_prompt(template_content, template_ctx))

        return rendered_prompts


if __name__ == "__main__":
    model = "openai:gpt-5-nano"
    agent_cfg = NativeAgentConfig(name="test_agent", model=model)
    print(agent_cfg)
