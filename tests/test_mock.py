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


# Path templates and schema composition for the mock server.
# A matching request returns the body registered for the winning route.
# A generated body is the value the schema itself requires: const, example,
# default, an enum member, or the declared JSON type.


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _required_node_followed(body: object) -> bool:
    """The node and its required child each carry an integer value.

    The child's own child is the cut-off and is not examined.
    """
    if not isinstance(body, dict):
        return False
    child = body.get("child")
    return (
        _is_json_int(body.get("value"))
        and isinstance(child, dict)
        and _is_json_int(child.get("value"))
    )


def _required_tree_followed(body: object) -> bool:
    """value is an integer and some kid was generated as a Tree.

    How many kids, and what those kids contain, is the cut-off.
    """
    if not isinstance(body, dict) or not _is_json_int(body.get("value")):
        return False
    kids = body.get("kids")
    if not isinstance(kids, list):
        return False
    return any(
        isinstance(kid, dict) and _is_json_int(kid.get("value"))
        for kid in kids
    )


def _required_mutual_followed(body: object) -> bool:
    """A and the required B each have a string name. B's link back is free."""
    if not isinstance(body, dict):
        return False
    other = body.get("b")
    return (
        isinstance(body.get("name"), str)
        and isinstance(other, dict)
        and isinstance(other.get("name"), str)
    )


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
    """Small search spec whose hit list refers to an object.

    That object's required rate_cents refers to an integer schema, and so do
    any additional properties. A loader that rejects the keyword fails here.
    The test does not require an extra property to be invented.
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
    """Load routes in a child process so a recursive schema cannot hang the test.

    A schema this small that does not finish within 5 seconds is treated as
    unbounded.
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
    if "content-length" in headers:
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
    """A `{name}` segment matches exactly one non-empty path segment.

    The response is the body registered on the route that wins. A request
    that matches no route returns 404 and names the path that was sent.
    """

    async def test_placeholder_matches_one_segment(self):
        """Both /invoices/123 and /invoices/456 return the /invoices/{id} body.

        Fails if the placeholder is ignored, or if only one hard-coded id matches.
        """
        routes = {"/invoices/{id}": {"GET": {"resource": "invoice"}}}
        async with _running(routes) as port:
            for target in ("/invoices/123", "/invoices/456"):
                status, body = await _exchange(port, "GET", target)
                assert status == 200
                assert body == {"resource": "invoice"}

    async def test_multiple_placeholders_each_match_one_segment(self):
        """Each placeholder is one segment, and a missing last segment does not match.

        Fails if one placeholder swallows several segments, or if the last
        placeholder is treated as optional.
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
        """A trailing slash is still the same template match.

        Fails if /invoices/123/ is rejected or sent to a different route.
        """
        routes = {"/invoices/{id}": {"GET": {"resource": "invoice"}}}
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/invoices/123/")
            assert status == 200
            assert body == {"resource": "invoice"}

    async def test_query_string_ignored_for_template(self):
        """A query string is ignored when matching a template.

        Fails if /invoices/123?page=1 does not hit /invoices/{id}.
        """
        routes = {"/invoices/{id}": {"GET": {"resource": "invoice"}}}
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/invoices/123?page=1&sort=asc")
            assert status == 200
            assert body == {"resource": "invoice"}

    async def test_literal_path_beats_template(self):
        """The literal /collections/wander/search wins over the collection template.

        The template is registered first, so a first-match implementation fails.
        Another collection id still uses the template. GET on the literal path
        is 404 because only POST is registered.

        Fails if wander gets the template body, or if every collection id gets
        the literal body.
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
        """GET /a/1/b uses /a/{x}/b, the template with more literal segments.

        The less specific template is registered first. /a/1/c does not have
        the literal segment b, so it uses /a/{x}/{y}.

        Fails if the two templates are treated as equal, or if a literal
        segment that does not match still captures the request.
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
        """An extra, missing, or empty segment, or the wrong method, is 404.

        The body is {"error": "not found", "path": <the path that was sent>}.
        None of these targets ends with a slash, so the path is unambiguous.

        Fails if a prefix, an empty placeholder, or the wrong method returns
        200, or if the 404 body leaves out the requested path.
        """
        routes = {
            "/invoices/{id}": {"GET": {"resource": "invoice"}},
            "/invoices/{id}/part": {"GET": {"resource": "part"}},
        }
        async with _running(routes) as port:
            status, body = await _exchange(port, method, target)
            assert status == 404
            assert body == {"error": "not found", "path": target}


class TestSpecTrailingSlash:
    """A trailing slash is optional on the spec path and on the request.

    `/x/{id}/` matches `/x/1` and `/x/1/`. `/foo/` matches `/foo` and `/foo/`.
    When both `/foo` and `/foo/` are registered, a request that matches one
    of them exactly gets that path's body. A matching path with no body for
    the request method is 404 and names the path that was sent.
    """

    async def test_template_spec_slash_matches_with_or_without_slash(self):
        """A template written with a trailing slash matches either request form.

        `/x` and `/x/` do not match, because the placeholder is still required.
        `/x/1/extra` does not match either. Fails if the slash on the spec
        path makes `/x/1` miss, or if a missing or extra segment is accepted.
        """
        routes = {"/x/{id}/": {"GET": {"resource": "item"}}}
        async with _running(routes) as port:
            for target in ("/x/1", "/x/1/"):
                status, body = await _exchange(port, "GET", target)
                assert status == 200
                assert body == {"resource": "item"}
            for target in ("/x", "/x/", "/x/1/extra"):
                status, body = await _exchange(port, "GET", target)
                assert status == 404
                assert body == {"error": "not found", "path": target}

    async def test_literal_spec_slash_matches_with_or_without_slash(self):
        """A literal path written with a trailing slash matches either request form.

        `/foo/bar` is a different path. Fails if `/foo` misses `/foo/`, or if
        a longer path is treated as the same route.
        """
        routes = {"/foo/": {"GET": {"resource": "foo"}}}
        async with _running(routes) as port:
            for target in ("/foo", "/foo/"):
                status, body = await _exchange(port, "GET", target)
                assert status == 200
                assert body == {"resource": "foo"}
            status, body = await _exchange(port, "GET", "/foo/bar")
            assert status == 404
            assert body == {"error": "not found", "path": "/foo/bar"}

    async def test_exact_literal_wins_when_both_slash_forms_exist(self):
        """Separate `/foo` and `/foo/` routes keep their own bodies.

        The slash form is registered first. A request equal to one of the
        two paths gets that path's body. Fails if the slash is stripped and
        both requests receive the same body.
        """
        routes = {
            "/foo/": {"GET": {"which": "slash"}},
            "/foo": {"GET": {"which": "plain"}},
        }
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/foo")
            assert status == 200
            assert body == {"which": "plain"}
            status, body = await _exchange(port, "GET", "/foo/")
            assert status == 200
            assert body == {"which": "slash"}

    async def test_wrong_method_on_slash_spec_is_404(self):
        """A slash spec that has no body for the request method is 404.

        GET still returns the registered body, so a matcher that rejects
        every request does not pass. Fails if POST is served as GET, or if
        the 404 body omits the path that was sent.
        """
        routes = {
            "/foo/": {"GET": {"resource": "foo"}},
            "/x/{id}/": {"GET": {"resource": "item"}},
        }
        async with _running(routes) as port:
            status, body = await _exchange(port, "GET", "/foo")
            assert status == 200
            assert body == {"resource": "foo"}
            for method, target in (
                ("POST", "/foo"),
                ("POST", "/foo/"),
                ("POST", "/x/1"),
                ("POST", "/x/1/"),
            ):
                status, body = await _exchange(port, method, target)
                assert status == 404
                assert body == {"error": "not found", "path": target}


class TestSchemaComposition:
    """Generated values come from the schema, not from a recorded response.

    A const, example, or default is that value. An enum value is one of its
    members. Otherwise the value has the declared JSON type. A `$ref` is
    replaced by the schema it names before that rule is applied.
    """

    def test_wander_search_resolves_nested_refs(self, tmp_path: Path):
        """The search template's hit list resolves nested schema references.

        POST /collections/{collectionId}/search returns hits, each an object
        whose required rate_cents refers to an integer schema. additionalProperties
        refers to that same integer, so a loader that rejects the keyword fails.
        An extra property does not have to be invented. Arrays are still
        non-empty, so an empty list cannot hide an unresolved item.

        Fails if the template key is missing, hits are missing, or rate_cents
        is an object instead of an integer.
        """
        routes = load_routes(_write_spec(tmp_path, _search_excerpt()))
        body = routes["/collections/{collectionId}/search"]["POST"]
        self._assert_search_body(body)

    async def test_wander_search_serves_resolved_body(self, tmp_path: Path):
        """The same search body is what the server returns for a wander request.

        The query string must not change which route matches. Fails if the
        server answers 404 for the template, or if the served rate_cents is
        not an integer.
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
        """A schema reference inside anyOf or oneOf is used, and null loses.

        Null is listed first. The other branch is the const string "Ada", so
        the body is exactly that string. Fails if the result is null, or if
        the reference is left as an empty object.
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
        """allOf of two schema references keeps both required constants.

        Fails if only the first branch is kept, or if a reference is unresolved
        and id or rate_cents is missing or is an object.
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
        """A response that is only a $ref returns that response's JSON example.

        The referenced response also has a schema that would build a different
        object. The example on the media type is the body instead. Fails if
        the response reference is dropped, or if the schema is generated in
        place of the example. parse_openapi and load_routes agree on that body.
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
        """The server returns the body of a response that was given by $ref.

        Fails if the loaded body followed the reference but the server returns
        something else, or answers 404 for the literal path.
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
        """A schema that refers to itself still loads as finite JSON.

        child is optional. value is required and is an integer, so an empty
        object does not pass. Fails if loading raises, does not finish within
        5 seconds, or returns a value that is not finite JSON.
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
        """Serving that self-referencing schema returns the same finite JSON.

        Fails if the server hangs or returns a body other than the one that
        was loaded.
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
        """Two schemas that refer to each other load as finite JSON.

        Fails if the cycle raises, does not finish within 5 seconds, or yields
        a value that is not finite JSON. The required name stays a string.
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

    async def test_required_self_reference_stays_finite(self, tmp_path: Path):
        """A required self-reference is followed, and loading still finishes.

        child is required, so the cycle cannot be skipped. value is an integer
        on the node and on that child. Nothing inside the child's own child
        is checked, so where expansion stops is not fixed.

        Fails if loading or serving does not finish within 5 seconds, raises,
        or leaves the child as an empty object.
        """
        spec = _openapi(
            {"/node": {"get": {"responses": _ok(
                {"$ref": "#/components/schemas/Node"}
            )}}},
            schemas={
                "Node": {
                    "type": "object",
                    "required": ["value", "child"],
                    "properties": {
                        "value": {"type": "integer"},
                        "child": {"$ref": "#/components/schemas/Node"},
                    },
                }
            },
        )
        await self._assert_required_recursion(
            tmp_path, spec, "required-self.json", "/node", "GET",
            _required_node_followed,
        )

    async def test_required_array_recursion_stays_finite(self, tmp_path: Path):
        """A required array of self-referencing items stays finite.

        kids is required, asks for at least two items, and each item is the
        same schema. That both follows the cycle and rejects an expansion
        that branches without bound. value is an integer, and at least one
        kid is a generated object with its own integer value. How many kids
        appear, and what is inside a kid's own kids, is not checked.

        Fails if loading or serving does not finish within 5 seconds, raises,
        or the array items are empty objects.
        """
        spec = _openapi(
            {"/tree": {"get": {"responses": _ok(
                {"$ref": "#/components/schemas/Tree"}
            )}}},
            schemas={
                "Tree": {
                    "type": "object",
                    "required": ["value", "kids"],
                    "properties": {
                        "value": {"type": "integer"},
                        "kids": {
                            "type": "array",
                            "minItems": 2,
                            "items": {"$ref": "#/components/schemas/Tree"},
                        },
                    },
                }
            },
        )
        await self._assert_required_recursion(
            tmp_path, spec, "required-tree.json", "/tree", "GET",
            _required_tree_followed,
        )

    async def test_required_mutual_recursion_stays_finite(self, tmp_path: Path):
        """Mutual recursion stays finite when both links are required.

        A must include B, and B must include A, so neither side can skip the
        other. name is a string on A and on that B. B's link back to A is
        not inspected.

        Fails if loading or serving does not finish within 5 seconds, raises,
        or the followed schema is an empty object.
        """
        spec = _openapi(
            {"/a": {"get": {"responses": _ok(
                {"$ref": "#/components/schemas/A"}
            )}}},
            schemas={
                "A": {
                    "type": "object",
                    "required": ["name", "b"],
                    "properties": {
                        "name": {"type": "string"},
                        "b": {"$ref": "#/components/schemas/B"},
                    },
                },
                "B": {
                    "type": "object",
                    "required": ["name", "a"],
                    "properties": {
                        "name": {"type": "string"},
                        "a": {"$ref": "#/components/schemas/A"},
                    },
                },
            },
        )
        await self._assert_required_recursion(
            tmp_path, spec, "required-mutual.json", "/a", "GET",
            _required_mutual_followed,
        )

    async def _assert_required_recursion(
        self, tmp_path, spec, filename, path, method, field_ok,
    ):
        routes = _load_bounded(_write_spec(tmp_path, spec, filename))
        loaded = routes[path][method]
        json.dumps(loaded)
        assert isinstance(loaded, dict)
        assert field_ok(loaded), loaded
        async with _running(routes) as port:
            status, served = await _exchange(port, method, path)
        assert status == 200
        assert served == loaded
        json.dumps(served)
        assert field_ok(served), served

    @pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
    def test_union_prefers_non_null_branch(self, keyword: str):
        """anyOf and oneOf prefer a string over null when null is listed first.

        The string branch is an enum, so an arbitrary string still fails.
        generate_instance is called with the schema alone. Fails if the result
        is null, or a string other than "alpha" or "beta".
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
        """An allOf object contains the required properties of every part.

        Fails if rate_cents from the second part is missing, or if a property
        declared as a string or integer is generated as an object.
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
        """A const schema generates that exact value, including from a spec file.

        Fails if const is ignored and some other value of that JSON type is
        invented, or if parse_openapi and load_routes disagree on the body.
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
        """example, default, and enum still apply after a schema reference.

        Each keyword sits on its own referenced schema, so they do not compete.
        The enum value may be any member; which member is not specified.
        The same keywords on an inline schema are already covered above.

        Fails if following the reference drops the keyword and emits some
        other string.
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
        """A media-type example is the body served for a template route.

        The schema would otherwise require a different constant. The route
        key stays written as /invoices/{id}. Fails if the placeholder route
        matches but the example is discarded, or if the key is rewritten
        without the placeholder.
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
