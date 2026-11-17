"""Tests for mock server generation."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from baton.mock import MockServer, generate_instance, load_routes, parse_openapi
from baton.tracing import NullExporter, parse_traceparent


class TestGenerateInstance:
    def test_string(self):
        result = generate_instance({"type": "string"})
        assert isinstance(result, str)
        assert len(result) >= 5

    def test_string_with_example(self):
        result = generate_instance({"type": "string", "example": "hello"})
        assert result == "hello"

    def test_integer(self):
        result = generate_instance({"type": "integer"})
        assert isinstance(result, int)

    def test_integer_range(self):
        result = generate_instance({"type": "integer", "minimum": 10, "maximum": 10})
        assert result == 10

    def test_number(self):
        result = generate_instance({"type": "number"})
        assert isinstance(result, float)

    def test_boolean(self):
        result = generate_instance({"type": "boolean"})
        assert result is True

    def test_array(self):
        result = generate_instance({"type": "array", "items": {"type": "integer"}})
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_object(self):
        result = generate_instance({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        })
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result

    def test_enum(self):
        result = generate_instance({"type": "string", "enum": ["a", "b", "c"]})
        assert result == "a"

    def test_default(self):
        result = generate_instance({"type": "string", "default": "fallback"})
        assert result == "fallback"

    def test_string_format_email(self):
        result = generate_instance({"type": "string", "format": "email"})
        assert "@" in result

    def test_string_format_date(self):
        result = generate_instance({"type": "string", "format": "date"})
        assert "2026" in result

    def test_null_type(self):
        result = generate_instance({"type": "null"})
        assert result is None


class TestParseOpenAPI:
    def test_parse_simple(self, tmp_path: Path):
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "name": {"type": "string"},
                                                },
                                            },
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
        }
        spec_path = tmp_path / "api.yaml"
        import yaml
        with open(spec_path, "w") as f:
            yaml.dump(spec, f)

        routes = parse_openapi(str(spec_path))
        assert "/users" in routes
        assert "GET" in routes["/users"]
        body = routes["/users"]["GET"]
        assert isinstance(body, list)

    def test_parse_with_ref(self, tmp_path: Path):
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/user": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/User"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "email": {"type": "string", "format": "email"},
                        },
                        "required": ["id", "email"],
                    }
                }
            },
        }
        spec_path = tmp_path / "api.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        routes = parse_openapi(str(spec_path))
        body = routes["/user"]["GET"]
        assert "id" in body
        assert "email" in body

    def test_parse_with_example(self, tmp_path: Path):
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/status": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "example": {"status": "ok", "version": "1.0"},
                                        "schema": {"type": "object"},
                                    }
                                }
                            }
                        }
                    }
                }
            },
        }
        spec_path = tmp_path / "api.yaml"
        import yaml
        with open(spec_path, "w") as f:
            yaml.dump(spec, f)

        routes = parse_openapi(str(spec_path))
        assert routes["/status"]["GET"] == {"status": "ok", "version": "1.0"}


class TestLoadRoutes:
    def test_load_openapi(self, tmp_path: Path):
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/health": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "example": {"ok": True}
                                    }
                                }
                            }
                        }
                    }
                }
            },
        }
        spec_path = tmp_path / "spec.yaml"
        import yaml
        with open(spec_path, "w") as f:
            yaml.dump(spec, f)

        routes = load_routes(str(spec_path))
        assert "/health" in routes

    def test_load_json_schema(self, tmp_path: Path):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        spec_path = tmp_path / "schema.json"
        with open(spec_path, "w") as f:
            json.dump(schema, f)

        routes = load_routes(str(spec_path))
        assert "/" in routes
        assert "GET" in routes["/"]

    def test_load_missing(self):
        routes = load_routes("/nonexistent/spec.yaml")
        assert routes == {}


class TestMockServer:
    async def test_start_and_stop(self):
        server = MockServer()
        server.add_default_routes(16001)
        await server.start()
        assert server.is_running
        await server.stop()
        assert not server.is_running

    async def test_serves_route(self):
        server = MockServer()
        server.add_routes(16002, {
            "/health": {"GET": {"status": "ok"}},
        })
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 16002)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks)
            writer.close()

            assert b"200 OK" in response
            body = response.split(b"\r\n\r\n", 1)[1]
            data = json.loads(body)
            assert data["status"] == "ok"
        finally:
            await server.stop()

    async def test_404_for_unknown_route(self):
        server = MockServer()
        server.add_routes(16003, {
            "/health": {"GET": {"status": "ok"}},
        })
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 16003)
            writer.write(b"GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks)
            writer.close()
            assert b"404" in response
        finally:
            await server.stop()

    async def test_multiple_ports(self):
        server = MockServer()
        server.add_routes(16004, {"/a": {"GET": {"port": "a"}}})
        server.add_routes(16005, {"/b": {"GET": {"port": "b"}}})
        await server.start()
        try:
            # Port A
            reader, writer = await asyncio.open_connection("127.0.0.1", 16004)
            writer.write(b"GET /a HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            resp_a = b"".join(chunks)
            writer.close()

            # Port B
            reader, writer = await asyncio.open_connection("127.0.0.1", 16005)
            writer.write(b"GET /b HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            resp_b = b"".join(chunks)
            writer.close()

            body_a = json.loads(resp_a.split(b"\r\n\r\n", 1)[1])
            body_b = json.loads(resp_b.split(b"\r\n\r\n", 1)[1])
            assert body_a["port"] == "a"
            assert body_b["port"] == "b"
        finally:
            await server.stop()

    async def test_query_string_ignored_in_route_matching(self):
        server = MockServer()
        server.add_routes(16013, {
            "/check": {"GET": {"available": True}},
        })
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 16013)
            writer.write(
                b"GET /check?unitId=00000000-0000-0000-0000-000000000000"
                b"&region=us-east-1&field=availability"
                b"&localFrom=2026-06-01&localTo=2026-06-07 HTTP/1.1\r\n"
                b"Host: localhost\r\n\r\n"
            )
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks)
            writer.close()

            assert b"200 OK" in response
            body = json.loads(response.split(b"\r\n\r\n", 1)[1])
            assert body["available"] is True
        finally:
            await server.stop()

    async def test_default_routes(self):
        server = MockServer()
        server.add_default_routes(16006)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 16006)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks)
            writer.close()
            assert b"200" in response
            body = json.loads(response.split(b"\r\n\r\n", 1)[1])
            assert body["status"] == "ok"
        finally:
            await server.stop()


class TestMockServerTracing:
    async def test_mock_emits_spans(self):
        """Mock server with span exporter creates spans."""
        exporter = NullExporter()
        server = MockServer(span_exporter=exporter, node_name="mock-api")
        server.add_routes(16010, {"/health": {"GET": {"status": "ok"}}})
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 16010)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            writer.close()
            assert b"200" in b"".join(chunks)

            spans = server.drain_spans()
            assert len(spans) == 1
            assert spans[0].node_name == "mock-api"
            assert spans[0].attributes["http.method"] == "GET"
            assert spans[0].attributes["http.path"] == "/health"
            assert spans[0].attributes["mock"] == "true"
        finally:
            await server.stop()

    async def test_mock_propagates_traceparent(self):
        """Mock server extracts traceparent from request."""
        exporter = NullExporter()
        server = MockServer(span_exporter=exporter, node_name="mock-svc")
        server.add_routes(16011, {"/api": {"POST": {"ok": True}}})
        await server.start()
        try:
            trace_id = "0af7651916cd43dd8448eb211c80319c"
            parent_span = "b7ad6b7169203331"
            tp = f"00-{trace_id}-{parent_span}-01"

            reader, writer = await asyncio.open_connection("127.0.0.1", 16011)
            writer.write(f"POST /api HTTP/1.1\r\nHost: localhost\r\ntraceparent: {tp}\r\n\r\n".encode())
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            writer.close()

            spans = server.drain_spans()
            assert len(spans) == 1
            assert spans[0].trace_id == trace_id
            assert spans[0].parent_span_id == parent_span
        finally:
            await server.stop()

    async def test_mock_no_spans_without_exporter(self):
        """Mock server without exporter creates no spans."""
        server = MockServer()
        server.add_routes(16012, {"/health": {"GET": {"ok": True}}})
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 16012)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            writer.close()

            spans = server.drain_spans()
            assert len(spans) == 0
        finally:
            await server.stop()


# Acceptance coverage for path templates and schema composition.
# Oracles are the lane brief and JSON Schema / OpenAPI semantics.
# Authority is normative unless a docstring says the legacy suite is the source.


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _is_json_int(value: object) -> bool:
    """JSON integers are not booleans. True is an int subclass in Python."""
    return isinstance(value, int) and not isinstance(value, bool)


def _openapi(
    paths: dict,
    schemas: dict | None = None,
    responses: dict | None = None,
) -> dict:
    document = {
        "openapi": "3.0.3",
        "info": {"title": "mock-contract", "version": "1"},
        "paths": paths,
    }
    components = {}
    if schemas is not None:
        components["schemas"] = schemas
    if responses is not None:
        components["responses"] = responses
    if components:
        document["components"] = components
    return document


def _ok(schema: dict | None = None, example: object | None = None) -> dict:
    media: dict = {}
    if schema is not None:
        media["schema"] = schema
    if example is not None:
        media["example"] = example
    return {
        "200": {
            "description": "ok",
            "content": {"application/json": media},
        }
    }


def _search_excerpt() -> dict:
    """Minimal stand-in for the current search path and a pricing-shaped hit.

    items is a schema ref to an object whose required rate_cents is an Int
    ref. additionalProperties is an Int ref so a resolver that rejects that
    keyword fails the load. The contract does not require an extra key.
    """
    return _openapi(
        {
            "/collections/{collectionId}/search": {
                "post": {
                    "responses": _ok({
                        "type": "object",
                        "required": ["hits"],
                        "properties": {
                            "hits": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/Hit"},
                            }
                        },
                    })
                }
            }
        },
        schemas={
            "Int": {"type": "integer"},
            "Hit": {
                "type": "object",
                "required": ["id", "rate_cents"],
                "properties": {
                    "id": {"type": "string"},
                    "rate_cents": {"$ref": "#/components/schemas/Int"},
                },
                "additionalProperties": {"$ref": "#/components/schemas/Int"},
            },
        },
    )


def _write_spec(tmp_path: Path, document: dict, name: str = "spec.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _load_bounded(spec_path: str, seconds: float = 5.0) -> dict:
    """Run load_routes in a child so a recursive schema cannot hang this process.

    The 5 second ceiling is the test's reading of "bounded time": a tiny schema
    that does not finish is not bounded. The contract states no byte limit.
    """
    script = (
        "import json, sys\n"
        "from baton.mock import load_routes\n"
        "json.dump(load_routes(sys.argv[1]), sys.stdout)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, spec_path],
            capture_output=True,
            text=True,
            timeout=seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"load_routes did not finish within {seconds}s") from exc
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout)


def _decode_chunked(body: bytes) -> bytes:
    chunks = []
    rest = body
    while rest:
        line, sep, rest = rest.partition(b"\r\n")
        if not sep:
            break
        size_token = line.split(b";", 1)[0].strip()
        if not size_token:
            break
        size = int(size_token, 16)
        if size == 0:
            break
        chunks.append(rest[:size])
        rest = rest[size:]
        if rest.startswith(b"\r\n"):
            rest = rest[2:]
    return b"".join(chunks)


def _parse_response(raw: bytes) -> tuple[int, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.lower()] = value.strip().lower()
    if headers.get("transfer-encoding") == "chunked":
        body = _decode_chunked(body)
    elif "content-length" in headers:
        body = body[: int(headers["content-length"])]
    return status, body


async def _exchange(port: int, method: str, target: str) -> tuple[int, object]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        message = (
            f"{method} {target} HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Connection: close\r\n"
        )
        if method in {"POST", "PUT", "PATCH"}:
            message += "Content-Length: 0\r\n"
        writer.write(f"{message}\r\n".encode())
        await writer.drain()
        chunks = []
        while True:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        writer.close()
        await writer.wait_closed()
    status, body = _parse_response(b"".join(chunks))
    return status, json.loads(body)


@asynccontextmanager
async def _running(routes: dict):
    server = MockServer()
    port = _free_port()
    server.add_routes(port, routes)
    await server.start()
    try:
        yield port
    finally:
        await server.stop()


class TestPathTemplates:
    """R1–R4. The registered body is the oracle: the request must return the
    body of the route the brief says wins, or the stated 404 object.
    """

    async def test_placeholder_matches_one_segment(self):
        """R1. GET /invoices/123 and /invoices/456 both return the /invoices/{id} body.

        Falsifier: the template is ignored, or only a hard-coded id matches.
        """
        routes = {"/invoices/{id}": {"GET": {"resource": "invoice"}}}
        async with _running(routes) as port:
            for target in ("/invoices/123", "/invoices/456"):
                status, body = await _exchange(port, "GET", target)
                assert status == 200
                assert body == {"resource": "invoice"}

    async def test_multiple_placeholders_each_match_one_segment(self):
        """R1. Each {param} is one non-empty segment. A missing segment does not match.

        Falsifier: one placeholder consumes several segments, or the last
        placeholder is optional.
        """
        routes = {"/orgs/{org}/invoices/{id}": {"GET": {"resource": "invoice"}}}
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/orgs/acme/invoices/9")
            assert status == 200
            assert body == {"resource": "invoice"}
            status, body = await _exchange(port, "GET", "/orgs/acme/invoices")
            assert status == 404
            assert body == {"error": "not found", "path": "/orgs/acme/invoices"}

    async def test_trailing_slash_still_matches_template(self):
        """R1. A trailing slash is tolerated and does not become an extra segment.

        Falsifier: /invoices/123/ is rejected or routed as a different path.
        """
        routes = {"/invoices/{id}": {"GET": {"resource": "invoice"}}}
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/invoices/123/")
            assert status == 200
            assert body == {"resource": "invoice"}

    async def test_query_string_ignored_for_template(self):
        """R1. The query string is not part of template matching.

        Falsifier: /invoices/123?page=1 misses /invoices/{id}.
        """
        routes = {"/invoices/{id}": {"GET": {"resource": "invoice"}}}
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/invoices/123?page=1&sort=asc")
            assert status == 200
            assert body == {"resource": "invoice"}

    async def test_literal_path_beats_template(self):
        """R2. /collections/wander/search wins over /collections/{collectionId}/search.

        The template is registered first, so first-match-wins fails this test.
        A different collection id still uses the template. The wrong method
        stays a 404 (R4).

        Falsifier: wander receives the template body, or every collection id
        receives the literal body.
        """
        routes = {
            "/collections/{collectionId}/search": {"POST": {"matched": "template"}},
            "/collections/wander/search": {"POST": {"matched": "literal"}},
        }
        async with _running(routes) as port:
            status, body = await _exchange(port, "POST", "/collections/wander/search")
            assert status == 200
            assert body == {"matched": "literal"}
            status, body = await _exchange(port, "POST", "/collections/other/search")
            assert status == 200
            assert body == {"matched": "template"}
            status, body = await _exchange(port, "GET", "/collections/wander/search")
            assert status == 404
            assert body == {"error": "not found", "path": "/collections/wander/search"}

    async def test_more_specific_template_wins(self):
        """R3. GET /a/1/b returns the body of /a/{x}/b, not /a/{x}/{y}.

        The less specific template is registered first. /a/1/c does not fit
        the literal segment, so it uses /a/{x}/{y}.

        Falsifier: the two templates are tied by length, or a non-matching
        literal segment still captures the request.
        """
        routes = {
            "/a/{x}/{y}": {"GET": {"matched": "two-params"}},
            "/a/{x}/b": {"GET": {"matched": "literal-b"}},
        }
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/a/1/b")
            assert status == 200
            assert body == {"matched": "literal-b"}
            status, body = await _exchange(port, "GET", "/a/1/c")
            assert status == 200
            assert body == {"matched": "two-params"}

    @pytest.mark.parametrize(
        ("method", "target"),
        [
            ("GET", "/invoices/123/extra"),
            ("GET", "/invoices"),
            ("GET", "/invoices//part"),
            ("POST", "/invoices/123"),
            ("GET", "/missing"),
        ],
    )
    async def test_non_match_is_json_404(self, method: str, target: str):
        """R4. Extra, missing, or empty segments, and the wrong method, are 404.

        The body is exactly {"error": "not found", "path": <request path>}.
        These targets have no trailing-slash ambiguity, so the request path
        is the origin-form target.

        Falsifier: a prefix match, an empty placeholder, or the wrong method
        returns 200, or the 404 body omits the path that was requested.
        """
        routes = {
            "/invoices/{id}": {"GET": {"resource": "invoice"}},
            "/invoices/{id}/part": {"GET": {"resource": "part"}},
        }
        async with _running(routes) as port:
            status, body = await _exchange(port, method, target)
            assert status == 404
            assert body == {"error": "not found", "path": target}


class TestSchemaComposition:
    """R5–R11. Expected values are the schema's const, example, default, or
    declared JSON type. Nothing here is recorded from a server run.
    """

    def test_wander_search_resolves_nested_refs(self, tmp_path: Path):
        """R5. A wander search request hits the template and returns resolved refs.

        Excerpt of the current search/pricing shape: POST
        /collections/{collectionId}/search, items $ref to an object whose
        required rate_cents is $ref Int. additionalProperties is also an Int
        ref so a resolver that rejects that keyword fails this load. The
        contract does not require a synthesized extra property.

        Falsifier: /collections/wander/search is 404, hits are missing, or
        rate_cents is an object instead of an integer. Legacy array behavior
        that still stands (R11) is a non-empty array, so an empty hits list
        cannot hide an unresolved item ref.
        """
        routes = load_routes(_write_spec(tmp_path, _search_excerpt()))
        body = routes["/collections/{collectionId}/search"]["POST"]
        self._assert_search_body(body)

    async def test_wander_search_serves_resolved_body(self, tmp_path: Path):
        """R5 at the HTTP boundary. Same excerpt, served by MockServer.

        The query string is the real search shape and must not affect matching.
        Falsifier: the loaded body is right but the server 404s the template,
        or the served rate_cents is not an integer.
        """
        routes = load_routes(_write_spec(tmp_path, _search_excerpt(), "search.json"))
        async with _running(routes) as port:
            status, body = await _exchange(
                port, "POST", "/collections/wander/search?page=1"
            )
        assert status == 200
        self._assert_search_body(body)

    @staticmethod
    def _assert_search_body(body: object) -> None:
        assert isinstance(body, dict)
        hits = body["hits"]
        assert isinstance(hits, list) and len(hits) >= 1
        for hit in hits:
            assert isinstance(hit, dict)
            assert isinstance(hit.get("id"), str)
            assert _is_json_int(hit.get("rate_cents"))

    @pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
    def test_ref_inside_union_prefers_value(self, tmp_path: Path, keyword: str):
        """R5 and R8. A $ref inside anyOf/oneOf is resolved, and null loses.

        Null is listed first. The other branch is const "Ada", so the body is
        exactly that string. First-branch and unresolved-ref both fail.

        Falsifier: the result is null, or an object standing in for the ref.
        """
        spec = _openapi(
            {"/name": {"get": {"responses": _ok({
                keyword: [
                    {"type": "null"},
                    {"$ref": "#/components/schemas/Name"},
                ],
            })}}},
            schemas={"Name": {"const": "Ada"}},
        )
        routes = load_routes(_write_spec(tmp_path, spec, f"{keyword}.json"))
        assert routes["/name"]["GET"] == "Ada"

    def test_ref_inside_allof_merges_required_fields(self, tmp_path: Path):
        """R5 and R9. allOf of two schema refs contains both required consts.

        Falsifier: only the first branch is kept, or a ref is left unresolved
        so id or rate_cents is missing or is an object.
        """
        spec = _openapi(
            {"/price": {"get": {"responses": _ok({
                "allOf": [
                    {"$ref": "#/components/schemas/WithId"},
                    {"$ref": "#/components/schemas/WithCents"},
                ],
            })}}},
            schemas={
                "WithId": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"const": "item"}},
                },
                "WithCents": {
                    "type": "object",
                    "required": ["rate_cents"],
                    "properties": {"rate_cents": {"const": 5}},
                },
            },
        )
        routes = load_routes(_write_spec(tmp_path, spec))
        body = routes["/price"]["GET"]
        assert isinstance(body, dict)
        assert body.get("id") == "item"
        assert body.get("rate_cents") == 5

    def test_response_ref_returns_its_json_body(self, tmp_path: Path):
        """R6 and R11. A response $ref returns that response's JSON example.

        The referenced response also has a schema that would generate a
        different object. The media-type example still wins.

        Falsifier: the response ref is dropped, or the schema is generated
        instead of the example.
        """
        spec = _openapi(
            {"/status": {"get": {"responses": {
                "200": {"$ref": "#/components/responses/Status"},
            }}}},
            responses={
                "Status": {
                    "description": "ok",
                    "content": {
                        "application/json": {
                            "example": {"status": "ok"},
                            "schema": {
                                "type": "object",
                                "required": ["other"],
                                "properties": {"other": {"const": "no"}},
                            },
                        }
                    },
                }
            },
        )
        path = _write_spec(tmp_path, spec)
        routes = load_routes(path)
        assert routes["/status"]["GET"] == {"status": "ok"}
        assert parse_openapi(path) == routes

    async def test_response_ref_is_served(self, tmp_path: Path):
        """R6 at HTTP. The caller receives the referenced response body.

        Falsifier: load_routes followed the ref but the server returns a
        different body, or 404s the literal path.
        """
        spec = _openapi(
            {"/status": {"get": {"responses": {
                "200": {"$ref": "#/components/responses/Status"},
            }}}},
            responses={
                "Status": {
                    "description": "ok",
                    "content": {"application/json": {"example": {"status": "ok"}}},
                }
            },
        )
        routes = load_routes(_write_spec(tmp_path, spec, "response.json"))
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/status")
        assert status == 200
        assert body == {"status": "ok"}

    def test_self_reference_loads_finite_json(self, tmp_path: Path):
        """R7. A schema that refers to itself loads as finite JSON.

        child is optional. value is a required integer, so an empty object
        does not pass. Falsifier: load_routes raises, does not finish within
        5 seconds, or returns a value json.dumps cannot encode.
        """
        spec = _openapi(
            {"/node": {"get": {"responses": _ok(
                {"$ref": "#/components/schemas/Node"}
            )}}},
            schemas={
                "Node": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {
                        "value": {"type": "integer"},
                        "child": {"$ref": "#/components/schemas/Node"},
                    },
                }
            },
        )
        routes = _load_bounded(_write_spec(tmp_path, spec))
        body = routes["/node"]["GET"]
        assert isinstance(body, dict)
        assert _is_json_int(body["value"])
        json.dumps(body)

    async def test_self_reference_http_body_is_finite(self, tmp_path: Path):
        """R7 served. The HTTP body is the finite JSON load_routes produced.

        Falsifier: the server hangs or returns a body that is not that JSON.
        """
        spec = _openapi(
            {"/node": {"get": {"responses": _ok(
                {"$ref": "#/components/schemas/Node"}
            )}}},
            schemas={
                "Node": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {
                        "value": {"type": "integer"},
                        "child": {"$ref": "#/components/schemas/Node"},
                    },
                }
            },
        )
        routes = _load_bounded(_write_spec(tmp_path, spec, "self.json"))
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/node")
        assert status == 200
        assert body == routes["/node"]["GET"]
        assert _is_json_int(body["value"])
        json.dumps(body)

    def test_mutual_reference_loads_finite_json(self, tmp_path: Path):
        """R7. Two schemas that refer to each other load in bounded time.

        Falsifier: the cycle raises, hangs past 5 seconds, or yields a value
        that is not finite JSON. The required name stays a string.
        """
        spec = _openapi(
            {"/a": {"get": {"responses": _ok(
                {"$ref": "#/components/schemas/A"}
            )}}},
            schemas={
                "A": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "b": {"$ref": "#/components/schemas/B"},
                    },
                },
                "B": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "a": {"$ref": "#/components/schemas/A"},
                    },
                },
            },
        )
        routes = _load_bounded(_write_spec(tmp_path, spec))
        body = routes["/a"]["GET"]
        assert isinstance(body, dict)
        assert isinstance(body.get("name"), str)
        json.dumps(body)

    @pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
    def test_union_prefers_non_null_branch(self, keyword: str):
        """R8. "string or null" yields a string, even when null is listed first.

        The string branch is an enum, so a generator that ignores the union
        and emits an arbitrary string still fails. Calling generate_instance
        with only the schema must keep working.

        Falsifier: the result is null, or a string outside the enum.
        """
        schema = {
            keyword: [
                {"type": "null"},
                {"type": "string", "enum": ["alpha", "beta"]},
            ]
        }
        generated = generate_instance(schema)
        assert generated == "alpha" or generated == "beta"

    def test_allof_contains_every_required_property(self):
        """R9. The object contains the required properties of every allOf part.

        Falsifier: the second part's rate_cents is missing, or a part's
        declared type is replaced with an object.
        """
        result = generate_instance({
            "allOf": [
                {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "string"}},
                },
                {
                    "type": "object",
                    "required": ["rate_cents"],
                    "properties": {"rate_cents": {"type": "integer"}},
                },
            ]
        })
        assert isinstance(result, dict)
        assert isinstance(result.get("id"), str)
        assert _is_json_int(result.get("rate_cents"))

    def test_const_is_the_generated_value(self, tmp_path: Path):
        """R10. The generated value equals the const, including through a spec.

        Falsifier: const is ignored and a value of that JSON type is invented,
        or parse_openapi and load_routes disagree on the fixed body.
        """
        assert generate_instance({"const": 7}) == 7
        assert generate_instance({"const": True}) is True
        assert generate_instance({"const": None}) is None
        assert generate_instance({"const": {"ready": True}}) == {"ready": True}
        spec = _openapi({
            "/obj": {"get": {"responses": _ok({"const": {"ready": True}})}},
        })
        path = _write_spec(tmp_path, spec)
        routes = load_routes(path)
        assert routes["/obj"]["GET"] == {"ready": True}
        assert parse_openapi(path) == routes

    def test_example_default_and_enum_survive_ref(self, tmp_path: Path):
        """R11. Schema example, default, and enum still win after a $ref.

        Each keyword is on its own referenced schema, so they do not compete.
        Enum asserts membership, not which member: the brief does not rank
        members. Inline examples of these keywords already stand in the
        legacy tests.

        Falsifier: resolving the ref drops the keyword and emits a free string.
        """
        spec = _openapi(
            {"/label": {"get": {"responses": _ok({
                "type": "object",
                "required": ["by_example", "by_default", "by_enum"],
                "properties": {
                    "by_example": {"$ref": "#/components/schemas/ByExample"},
                    "by_default": {"$ref": "#/components/schemas/ByDefault"},
                    "by_enum": {"$ref": "#/components/schemas/ByEnum"},
                },
            })}}},
            schemas={
                "ByExample": {"type": "string", "example": "Ada"},
                "ByDefault": {"type": "string", "default": "fallback"},
                "ByEnum": {"type": "string", "enum": ["a", "b", "c"]},
            },
        )
        body = load_routes(_write_spec(tmp_path, spec))["/label"]["GET"]
        assert body.get("by_example") == "Ada"
        assert body.get("by_default") == "fallback"
        enum_value = body.get("by_enum")
        assert enum_value == "a" or enum_value == "b" or enum_value == "c"

    async def test_media_example_wins_on_template_path(self, tmp_path: Path):
        """R11. A media-type example is the served body of a template route.

        The schema would otherwise require a different const. The path key
        stays written as /invoices/{id}.

        Falsifier: template matching works but the example is discarded, or
        the route key is rewritten without the placeholder.
        """
        spec = _openapi({
            "/invoices/{id}": {"get": {"responses": _ok(
                schema={
                    "type": "object",
                    "required": ["generated"],
                    "properties": {"generated": {"const": "no"}},
                },
                example={"invoice": "kept"},
            )}},
        })
        path = _write_spec(tmp_path, spec)
        routes = load_routes(path)
        assert "/invoices/{id}" in routes
        assert routes["/invoices/{id}"]["GET"] == {"invoice": "kept"}
        assert parse_openapi(path) == routes
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/invoices/123")
        assert status == 200
        assert body == {"invoice": "kept"}
