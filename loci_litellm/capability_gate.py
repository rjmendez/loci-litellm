"""Capability gate: keep requests that need a capability off deployments that lack it.

A request that carries tool definitions or tool turns gets a `!<tag>` request tag for each tag in
LOCI_GATE_TOOLS_EXCLUDE_TAGS; a request with image content parts gets one for each tag in
LOCI_GATE_IMAGES_EXCLUDE_TAGS (comma-separated). With router tag filtering on and the unsuitable
deployments tagged, the router then skips those deployments.

No tags are configured by default, so the gate does nothing until you decide which deployments lack which
capability. It needs `router_settings.enable_tag_filtering: true` and matching `tags:` on the deployments.
LOCI_CAPABILITY_GATE_DISABLED=1 turns it off. Fail-open: an error is logged at WARNING and the request
proceeds unchanged. It never changes the model or the request body.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("loci_litellm.capability_gate")

_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _tags(env: str) -> list[str]:
    return [t.strip() for t in os.environ.get(env, "").split(",") if t.strip()]


def _disabled() -> bool:
    return os.environ.get("LOCI_CAPABILITY_GATE_DISABLED", "").strip().lower() in ("1", "true", "yes", "on")


def needs_tools(data: dict) -> bool:
    if data.get("tools") or data.get("functions"):
        return True
    for message in data.get("messages") or []:
        if isinstance(message, dict) and (message.get("role") in ("tool", "function") or message.get("tool_calls")):
            return True
    return False


def needs_images(data: dict) -> bool:
    for message in data.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES for part in content
        ):
            return True
    return False


_RULES = (
    ("tools", "LOCI_GATE_TOOLS_EXCLUDE_TAGS", needs_tools),
    ("images", "LOCI_GATE_IMAGES_EXCLUDE_TAGS", needs_images),
)


class LociCapabilityGate(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Any:
        if _disabled() or not isinstance(data, dict):
            return None
        try:
            exclude: list[str] = []
            triggered: list[str] = []
            for capability, env, detect in _RULES:
                tags = _tags(env)
                if tags and detect(data):
                    triggered.append(capability)
                    exclude.extend(f"!{t}" for t in tags)
            if not triggered:
                return None
            metadata = data.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            tags = [t for t in (metadata.get("tags") or []) if isinstance(t, str)]
            tags.extend(t for t in dict.fromkeys(exclude) if t not in tags)
            metadata["tags"] = tags
            metadata["loci_capability_gate"] = ",".join(triggered)
            data["metadata"] = metadata
        except Exception as exc:
            logger.warning("loci capability gate: fail-open: %r", exc)
        return None


loci_capability_gate = LociCapabilityGate()
