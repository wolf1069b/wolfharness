"""Session-isolated compression block store.

Provides an in-memory store that groups compression blocks by session,
supports parent-chain traversal, and enforces a per-session capacity limit.

The store accepts a ``session_id`` parameter for namespace isolation,
ensuring that compression blocks from different sessions are never
interleaved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from wolfharness.capabilities.dcp.state import CompressionBlock


@dataclass
class BlockStoreStats:
    """Aggregate statistics about compression blocks in a session.

    Attributes:
        total_blocks: Number of blocks stored for the session.
        algorithms_used: Number of blocks per compression kind (strategy).
        average_ratio: Average compressed content length across all blocks.
    """

    total_blocks: int
    algorithms_used: dict[str, int]
    average_ratio: float


#: Maximum depth of parent-chain traversal in ``get_chain``.
MAX_CHAIN_DEPTH: int = 10


class CompressionBlockStore:
    """Session-isolated in-memory storage for compression blocks.

    Each session (identified by a string key) maintains its own ordered list
    of ``CompressionBlock`` instances.  When a session exceeds
    ``MAX_BLOCKS_PER_SESSION`` the oldest block is evicted.

    The store uses the real ``session_id`` passed to each method for
    namespace isolation — no hardcoded fallback keys.

    Attributes:
        MAX_BLOCKS_PER_SESSION: Maximum number of blocks retained per session.
    """

    MAX_BLOCKS_PER_SESSION: int = 1000

    def __init__(self) -> None:
        """Initialise an empty block store."""
        self._sessions: dict[str, list[CompressionBlock]] = {}

    def put(self, session_id: str, block: CompressionBlock) -> str:
        """Store a block for *session_id*.

        If the session already has ``MAX_BLOCKS_PER_SESSION`` blocks the
        oldest (first inserted) block is evicted.

        Args:
            session_id: The session to store the block under.
            block: The compression block to store.

        Returns:
            The ``block_id`` of the stored block.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        session = self._sessions[session_id]
        session.append(block)
        if len(session) > self.MAX_BLOCKS_PER_SESSION:
            session.pop(0)
        return block.block_id

    def get(self, session_id: str, block_id: str) -> CompressionBlock | None:
        """Retrieve a block by ID for a given session.

        Args:
            session_id: The session to look up.
            block_id: The block identifier to find.

        Returns:
            The matching block, or ``None`` if the session or block doesn't
            exist.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        for block in session:
            if block.block_id == block_id:
                return block
        return None

    def get_chain(self, session_id: str, block_id: str) -> list[CompressionBlock]:
        """Follow the parent chain from *block_id* up to ``MAX_CHAIN_DEPTH`` levels deep.

        The returned list starts with *block_id* and follows
        ``parent_block_id`` references.  If *block_id* does not exist in the
        session an empty list is returned.

        Args:
            session_id: The session to look up.
            block_id: The starting block identifier.

        Returns:
            A list of blocks from child to ancestor (maximum ``MAX_CHAIN_DEPTH`` entries).
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []
        # Build a lookup for fast parent traversal.
        by_id: dict[str, CompressionBlock] = {b.block_id: b for b in session}
        if block_id not in by_id:
            return []
        result: list[CompressionBlock] = []
        current_id: str | None = block_id
        while current_id is not None and len(result) < MAX_CHAIN_DEPTH:
            block = by_id.get(current_id)
            if block is None:
                break
            result.append(block)
            current_id = block.parent_block_id
        return result

    def get_all(self, session_id: str) -> list[CompressionBlock]:
        """Return all blocks for a session.

        Args:
            session_id: The session to look up.

        Returns:
            A list of all blocks in the session, or an empty list if the
            session doesn't exist.
        """
        return self._sessions.get(session_id, [])

    def get_stats(self, session_id: str) -> BlockStoreStats:
        """Return aggregate statistics for a session's blocks.

        Args:
            session_id: The session to analyse.

        Returns:
            A ``BlockStoreStats`` instance with the computed metrics.
        """
        session = self._sessions.get(session_id, [])
        if not session:
            return BlockStoreStats(
                total_blocks=0,
                algorithms_used={},
                average_ratio=0.0,
            )
        total = len(session)
        algorithms: dict[str, int] = {}
        total_compressed_len = 0
        for block in session:
            algorithms[block.kind] = algorithms.get(block.kind, 0) + 1
            total_compressed_len += len(block.compressed_content)
        average_ratio = total_compressed_len / total
        return BlockStoreStats(
            total_blocks=total,
            algorithms_used=algorithms,
            average_ratio=average_ratio,
        )
