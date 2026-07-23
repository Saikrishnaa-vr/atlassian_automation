"""Reusable LangGraph checkpoint serializer for Pydantic v2 models.

LangGraph's default msgpack encoder calls ``model_dump()`` on Pydantic models,
which includes ``@computed_field`` values. On decode, those extra keys hit
``extra="forbid"`` validation and raise ``ValidationError``. The fallback is
``model_construct(**kwargs)``, which bypasses field coercion and leaves nested
models as plain dicts — breaking model validators that access attributes on
nested objects.

This module provides a serializer that excludes computed fields from the encoded
payload so that ``cls(**kwargs)`` always succeeds on the decode side, making
``model_construct`` unnecessary. It is the standard checkpointer for every
LangGraph workflow in this project.

Usage::

    from src.infrastructure.checkpoint_serializer import make_memory_saver

    compiled = graph.compile(
        checkpointer=make_memory_saver(),
        interrupt_before=["approval_gate"],
    )

Private LangGraph symbols pinned to LangGraph 1.2.9
----------------------------------------------------
- ``langgraph.checkpoint.serde.jsonplus.EXT_PYDANTIC_V2`` (int = 5)
- ``langgraph.checkpoint.serde.jsonplus._msgpack_default``
- ``langgraph.checkpoint.serde.jsonplus._option``

These symbols have been stable across the LangGraph 1.x series. If a LangGraph
upgrade breaks them, update this module in one place rather than hunting across
workflow files.
"""

from __future__ import annotations

from typing import Any

import ormsgpack
from pydantic import BaseModel as _BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import (
    EXT_PYDANTIC_V2 as _EXT_PYDANTIC_V2,
    JsonPlusSerializer as _JsonPlusSerializer,
    _msgpack_default as _lg_msgpack_default,
    _option as _lg_msgpack_option,
)


def _pydantic_no_computed_default(obj: Any) -> Any:
    """ormsgpack ``default`` handler that strips computed fields before encoding.

    Encodes Pydantic v2 model instances into the LangGraph ``EXT_PYDANTIC_V2``
    msgpack extension type, but excludes any ``@computed_field`` values from the
    payload.  This prevents the extra-key ``ValidationError`` that would otherwise
    force LangGraph's decoder to fall back to ``model_construct()``.

    For non-Pydantic objects, delegates to LangGraph's own default handler so
    that UUIDs, datetimes, enums, and other standard types are handled correctly.
    """
    if isinstance(obj, _BaseModel):
        computed = set(type(obj).model_computed_fields.keys())
        # Build kwargs from live field values rather than model_dump() so that
        # ormsgpack sees nested Pydantic instances and calls this handler on
        # each one.  model_dump() flattens everything to dicts recursively,
        # which means computed fields on nested models (e.g. StoryCollection.count
        # inside WorkflowState) are baked into those nested dicts and survive
        # into the checkpoint payload — triggering extra="forbid" ValidationError
        # and the model_construct fallback on decode.
        kwargs = {k: getattr(obj, k) for k in type(obj).model_fields if k not in computed}
        return ormsgpack.Ext(
            _EXT_PYDANTIC_V2,
            ormsgpack.packb(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    kwargs,
                    "model_validate_json",
                ),
                default=_pydantic_no_computed_default,
                option=_lg_msgpack_option,
            ),
        )
    return _lg_msgpack_default(obj)


class LangGraphCheckpointSerde(_JsonPlusSerializer):
    """Checkpoint serializer that strips Pydantic computed fields before encoding.

    Subclasses :class:`~langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`
    and overrides only ``dumps_typed`` to swap in
    :func:`_pydantic_no_computed_default` as the msgpack ``default`` handler.
    All deserialization logic is inherited unchanged from the parent — the decode
    path is unaffected because computed fields were never present in the payload.

    This is the standard serializer for every LangGraph workflow in this project.
    Pass it via :func:`make_memory_saver` rather than constructing it directly.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None:
            return "null", b""
        if isinstance(obj, bytes):
            return "bytes", obj
        if isinstance(obj, bytearray):
            return "bytearray", obj
        return "msgpack", ormsgpack.packb(
            obj, default=_pydantic_no_computed_default, option=_lg_msgpack_option
        )


def make_memory_saver() -> MemorySaver:
    """Return a ``MemorySaver`` wired with :class:`LangGraphCheckpointSerde`.

    This is the canonical way to create a checkpointer for a LangGraph workflow
    in this project.  Call it once per compiled graph::

        compiled = graph.compile(checkpointer=make_memory_saver(), ...)
    """
    return MemorySaver(serde=LangGraphCheckpointSerde())
