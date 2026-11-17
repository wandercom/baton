"""Mock server generation from contract specs.

Generates asyncio HTTP servers that serve valid responses for endpoints
defined in OpenAPI or JSON Schema specs. A single MockServer can serve
multiple ports simultaneously (for collapsed circuits).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import string
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from baton.tracing import SpanExporter

logger = logging.getLogger(__name__)


def _resolve_ref(
    value: dict, components: dict, seen: frozenset[str] = frozenset(),
    diagnostics: set[str] | None = None,
) -> dict:
    """Follow local component references, including response aliases."""
    if diagnostics is None:
        diagnostics = set()
    ref = value.get("$ref")
    if not ref:
        return value
    if ref in seen:
        return {}
    target: Any = {"components": components}
    if isinstance(ref, str) and ref.startswith("#/components/"):
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            target = target.get(part) if isinstance(target, dict) else None
    else:
        target = None
    if not isinstance(target, dict):
        if ref not in diagnostics:
            logger.debug("Unresolved mock reference: %s", ref)
            diagnostics.add(ref)
        target = {}
    resolved = _resolve_ref(target, components, seen | {ref}, diagnostics)
    siblings = {key: val for key, val in value.items() if key != "$ref"}
    return _merge_schema(resolved, siblings, diagnostics)


def _merge_schema(left: dict, right: dict, diagnostics: set[str] | None = None) -> dict:
    """Combine constraints for an allOf intersection."""
    merged = {**left, **right}
    if "allOf" in left and "allOf" in right:
        merged["allOf"] = left["allOf"] + right["allOf"]
    if ("enum" in left or "const" in left) and ("enum" in right or "const" in right):
        left_values = [left["const"]] if "const" in left else left["enum"]
        right_values = [right["const"]] if "const" in right else right["enum"]
        values = [value for value in left_values if value in right_values]
        if not values:
            message = "Empty mock enum intersection; using a value from the first part"
            if diagnostics is None or message not in diagnostics:
                logger.debug(message)
                if diagnostics is not None:
                    diagnostics.add(message)
            values = left_values or right_values or [None]
        merged["enum"] = values
        if "const" in merged:
            merged["const"] = values[0]
    if "properties" in left or "properties" in right:
        props = dict(left.get("properties", {}))
        for name, prop in right.get("properties", {}).items():
            props[name] = {"allOf": [props[name], prop]} if name in props else prop
        merged["properties"] = props
    if "required" in left or "required" in right:
        merged["required"] = list(
            dict.fromkeys(left.get("required", []) + right.get("required", []))
        )
    for key in ("minimum", "minLength", "minItems", "minProperties"):
        if key in left and key in right:
            merged[key] = max(left[key], right[key])
    for key in ("maximum", "maxLength", "maxItems", "maxProperties"):
        if key in left and key in right:
            merged[key] = min(left[key], right[key])
    return merged


def _expand_schema(
    schema: dict, components: dict, depth: int = 0,
    seen: frozenset[str] = frozenset(), diagnostics: set[str] | None = None,
) -> dict:
    ref = schema.get("$ref")
    if depth >= 32 or ref in seen:
        return {}
    schema = _resolve_ref(schema, components, seen, diagnostics)
    if ref:
        seen = seen | {ref}
    if "allOf" in schema:
        merged = {}
        for part in schema["allOf"]:
            expanded = _expand_schema(part, components, depth + 1, seen, diagnostics)
            merged = _merge_schema(merged, expanded, diagnostics)
        schema = _merge_schema(
            merged, {k: v for k, v in schema.items() if k != "allOf"}, diagnostics,
        )
    return schema


def generate_instance(
    schema: dict, components: dict | None = None, *,
    _depth: int = 0, _seen: frozenset[str] = frozenset(),
    _last_ref: str | None = None, _self_repeated: bool = False,
    _diagnostics: set[str] | None = None,
) -> Any:
    """Generate a random valid instance from a JSON Schema."""
    components = components or {}
    if _diagnostics is None:
        _diagnostics = set()
    ref = schema.get("$ref")
    repeated = ref in _seen
    # Emit one full child for direct self-reference, then cut its next link.
    # A link back through another schema remains a mutual-recursion cutoff.
    if _depth >= 32 or (repeated and (ref != _last_ref or _self_repeated)):
        return {}
    schema = _expand_schema(
        schema, components, seen=_seen - {ref} if repeated else _seen,
        diagnostics=_diagnostics,
    )
    if ref:
        _seen = _seen | {ref}
        _last_ref = ref
        _self_repeated = repeated
    for key in ("example", "default", "enum", "const"):
        if key in schema:
            return schema[key][0] if key == "enum" else schema[key]

    def generate(child: dict) -> Any:
        return generate_instance(
            child, components, _depth=_depth + 1, _seen=_seen,
            _last_ref=_last_ref, _self_repeated=_self_repeated,
            _diagnostics=_diagnostics,
        )

    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            base = {k: v for k, v in schema.items() if k != keyword}
            branches = [
                _merge_schema(base, branch, _diagnostics) for branch in schema[keyword]
            ]
            # A reference may itself describe a null branch. Try all non-null
            # results before accepting null, keeping declaration order otherwise.
            for branch in branches:
                value = generate(branch)
                if value is not None:
                    return value
            return None

    schema_type = schema.get("type", "object")
    if isinstance(schema_type, list):
        schema_type = next((kind for kind in schema_type if kind != "null"), "null")

    if schema_type == "string":
        fmt = schema.get("format", "")
        if fmt == "date":
            return "2026-01-01"
        if fmt == "date-time":
            return "2026-01-01T00:00:00Z"
        if fmt == "email":
            return "test@example.com"
        if fmt == "uri" or fmt == "url":
            return "https://example.com"
        length = schema.get("minLength", 5)
        return "".join(random.choices(string.ascii_lowercase, k=length))

    if schema_type == "integer":
        lo = int(schema.get("minimum", 1))
        hi = int(schema.get("maximum", 100))
        return random.randint(lo, hi)

    if schema_type == "number":
        return round(random.uniform(0, 100), 2)

    if schema_type == "boolean":
        return True

    if schema_type == "array":
        items = schema.get("items", {"type": "string"})
        count = schema.get("minItems", 1)
        return [generate(items) for _ in range(count)]

    if schema_type == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", list(props.keys())))
        obj = {}
        for name, prop_schema in props.items():
            if name in required:
                obj[name] = generate(prop_schema)
        additional = schema.get("additionalProperties")
        if not props and isinstance(additional, dict):
            for index in range(schema.get("minProperties", 0)):
                obj[f"key{index + 1}"] = generate(additional)
        return obj

    return None


def parse_openapi(spec_path: str) -> dict[str, dict[str, Any]]:
    """Parse an OpenAPI 3.x spec and return route table.

    Returns: {path: {METHOD: response_body}}
    """
    path = Path(spec_path)
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            spec = yaml.safe_load(f)
        else:
            spec = json.load(f)

    if not spec or "paths" not in spec:
        return {}

    components = spec.get("components", {})
    diagnostics: set[str] = set()
    routes: dict[str, dict[str, Any]] = {}

    for path_str, methods in spec["paths"].items():
        routes[path_str] = {}
        for method, operation in methods.items():
            if method.startswith("x-") or method == "parameters":
                continue
            method = method.upper()

            # Find the success response
            responses = operation.get("responses", {})
            success_resp = responses.get("200") or responses.get("201") or {}
            success_resp = _resolve_ref(success_resp, components, diagnostics=diagnostics)
            content = success_resp.get("content", {})
            json_content = content.get("application/json", {})
            resp_schema = json_content.get("schema", {})

            # Generate response body
            if "example" in json_content:
                body = json_content["example"]
            elif "example" in resp_schema:
                body = resp_schema["example"]
            elif resp_schema:
                body = generate_instance(resp_schema, components, _diagnostics=diagnostics)
            else:
                body = {}

            routes[path_str][method] = body

    return routes


def parse_json_schema(spec_path: str) -> dict[str, dict[str, Any]]:
    """Parse a JSON Schema and return a single-endpoint route table."""
    with open(spec_path) as f:
        schema = json.load(f)

    body = generate_instance(schema)
    return {"/": {"GET": body}}


def load_routes(spec_path: str) -> dict[str, dict[str, Any]]:
    """Load routes from a spec file (auto-detect format)."""
    path = Path(spec_path)
    if not path.exists():
        logger.warning(f"Contract spec not found: {spec_path}")
        return {}

    # Try OpenAPI first
    try:
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    except Exception:
        return {}

    if isinstance(data, dict) and ("openapi" in data or "paths" in data):
        return parse_openapi(spec_path)
    elif isinstance(data, dict) and ("type" in data or "properties" in data):
        return parse_json_schema(spec_path)
    else:
        return {}


class MockServer:
    """In-process mock HTTP server serving canned responses.

    Can serve routes for multiple ports simultaneously (collapsed mode).
    """

    def __init__(
        self,
        span_exporter: "SpanExporter | None" = None,
        node_name: str = "",
    ):
        self._route_tables: dict[int, dict[str, dict[str, Any]]] = {}
        self._templates: dict[int, list[tuple[re.Pattern, dict[str, Any]]]] = {}
        self._servers: list[asyncio.Server] = []
        self._span_exporter = span_exporter
        self._node_name = node_name
        self._span_buffer: list = []

    def add_routes(self, port: int, routes: dict[str, dict[str, Any]]) -> None:
        """Register routes for a port."""
        self._route_tables[port] = routes
        templates = []
        for path, methods in routes.items():
            segments = path.rstrip("/").split("/")
            parameters = [
                bool(re.fullmatch(r"\{[^{} /]+\}", segment)) for segment in segments
            ]
            if any(parameters):
                pattern = "/".join(
                    "[^/]+" if parameter else re.escape(segment)
                    for segment, parameter in zip(segments, parameters)
                )
                specificity = sum(
                    bool(segment) and not parameter
                    for segment, parameter in zip(segments, parameters)
                )
                templates.append((specificity, re.compile(pattern), methods))
        templates.sort(key=lambda entry: entry[0], reverse=True)
        self._templates[port] = [(pattern, methods) for _, pattern, methods in templates]

    def add_default_routes(self, port: int) -> None:
        """Add a default health endpoint for a port with no contract."""
        self.add_routes(port, {
            "/": {"GET": {"status": "mock", "port": port}},
            "/health": {"GET": {"status": "ok"}},
        })

    async def start(self, host: str = "127.0.0.1") -> None:
        """Start HTTP servers for all registered ports."""
        for port, routes in self._route_tables.items():
            server = await asyncio.start_server(
                lambda r, w, p=port: self._handle(r, w, p),
                host,
                port,
            )
            self._servers.append(server)
            logger.info(f"Mock server listening on {host}:{port} ({len(routes)} routes)")

    async def stop(self) -> None:
        for s in self._servers:
            s.close()
            await s.wait_closed()
        self._servers.clear()

    @property
    def is_running(self) -> bool:
        return len(self._servers) > 0

    def drain_spans(self) -> list:
        """Return and clear the span buffer."""
        drained = self._span_buffer[:]
        self._span_buffer.clear()
        return drained

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        port: int,
    ) -> None:
        """Handle an HTTP request: match route, return canned response."""
        try:
            start = time.monotonic()
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                writer.close()
                return

            # Read remaining headers and extract traceparent
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    key, val = line.split(b":", 1)
                    headers[key.strip().decode("ascii", errors="replace").lower()] = (
                        val.strip().decode("ascii", errors="replace")
                    )

            parts = request_line.decode("ascii", errors="replace").strip().split()
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) > 1 else "/"
            if "?" in path:
                path = path.split("?")[0]

            routes = self._route_tables.get(port, {})
            alt_path = path.rstrip("/") if path != "/" else path
            if path in routes:
                route = routes[path]
            elif alt_path in routes:
                route = routes[alt_path]
            elif path != "/" and alt_path + "/" in routes:
                route = routes[alt_path + "/"]
            else:
                route = {}
                for pattern, methods in self._templates.get(port, []):
                    if pattern.fullmatch(path) or pattern.fullmatch(alt_path):
                        route = methods
                        break
            body = route.get(method)

            status_code = 200
            if body is None:
                status_code = 404
                resp_body = json.dumps({"error": "not found", "path": path}).encode()
                writer.write(
                    f"HTTP/1.1 404 Not Found\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(resp_body)}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n".encode("ascii")
                )
                writer.write(resp_body)
            else:
                resp_body = json.dumps(body).encode()
                writer.write(
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(resp_body)}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n".encode("ascii")
                )
                writer.write(resp_body)

            await writer.drain()

            # Create span if exporter is configured
            if self._span_exporter:
                from baton.tracing import (
                    SpanData,
                    generate_span_id,
                    generate_trace_id,
                    parse_traceparent,
                )

                parent_ctx = parse_traceparent(headers.get("traceparent", ""))
                trace_id = parent_ctx.trace_id if parent_ctx else generate_trace_id()
                span_id = generate_span_id()
                span = SpanData(
                    name=f"mock_{self._node_name}_{method}",
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_ctx.span_id if parent_ctx else "",
                    start_time_ns=int(start * 1e9),
                    end_time_ns=int(time.monotonic() * 1e9),
                    attributes={
                        "http.method": method,
                        "http.path": path,
                        "http.status_code": str(status_code),
                        "mock": "true",
                    },
                    status="ok" if status_code < 400 else "error",
                    node_name=self._node_name,
                )
                self._span_buffer.append(span)

        except Exception as e:
            logger.debug(f"Mock server error: {e}")
        finally:
            writer.close()
