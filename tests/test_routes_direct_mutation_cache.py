"""
Regression tests for GitHub issue #16301.

Direct mutation of ``APIRouter.routes`` (via ``.append()``, ``.remove()``,
``.clear()``, etc.) must correctly invalidate the ``_IncludedRouter``
effective-candidates cache so that subsequent requests see the updated route
list rather than stale/ghost routes.

Each test follows the same structure:
1. Build a child router, include it in a parent router / app.
2. Dispatch at least one request so that the cache is warmed up.
3. Mutate ``child.routes`` directly.
4. Assert that a subsequent request reflects the mutation.
"""

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_route(path: str, return_value: str) -> APIRoute:
    """Create an APIRoute without going through the router decorator."""

    def endpoint() -> str:  # pragma: no cover
        return return_value

    endpoint.__name__ = f"endpoint_{path.strip('/').replace('/', '_')}"
    return APIRoute(path, endpoint=endpoint, methods=["GET"])


# ---------------------------------------------------------------------------
# Test: direct .append() on child.routes is visible after cache warm-up
# ---------------------------------------------------------------------------


def test_direct_append_to_child_routes_invalidates_cache():
    """
    After the cache has been populated by a first request, appending a route
    directly to child.routes must be visible on subsequent requests.
    """
    child = APIRouter()

    @child.get("/a")
    def route_a() -> str:  # pragma: no cover
        return "a"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up the cache.
    resp = client.get("/v1/a")
    assert resp.status_code == 200

    # Direct mutation: add a new route without going through add_api_route.
    new_route = _make_route("/b", "b")
    child.routes.append(new_route)

    # The cache must be invalidated; the new route must be reachable.
    resp2 = client.get("/v1/b")
    assert resp2.status_code == 200, (
        f"Expected 200 after direct .append(), got {resp2.status_code}. "
        "Cache was not invalidated."
    )


# ---------------------------------------------------------------------------
# Test: direct .remove() on child.routes makes the route unreachable
# ---------------------------------------------------------------------------


def test_direct_remove_from_child_routes_invalidates_cache():
    """
    After the cache has been populated by a first request, removing a route
    directly from child.routes must make subsequent requests return 404.
    """
    child = APIRouter()

    @child.get("/gone")
    def route_gone() -> str:  # pragma: no cover
        return "gone"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up the cache and confirm the route is reachable.
    resp = client.get("/v1/gone")
    assert resp.status_code == 200

    # Direct mutation: remove the route without going through any API method.
    route_to_remove = child.routes[0]
    child.routes.remove(route_to_remove)

    # The cache must be invalidated; the removed route must now return 404.
    resp2 = client.get("/v1/gone")
    assert resp2.status_code == 404, (
        f"Expected 404 after direct .remove(), got {resp2.status_code}. "
        "Cache was not invalidated (ghost route remains)."
    )


# ---------------------------------------------------------------------------
# Test: direct .clear() on child.routes removes all routes
# ---------------------------------------------------------------------------


def test_direct_clear_of_child_routes_invalidates_cache():
    """
    Clearing child.routes directly must make all routes return 404.
    """
    child = APIRouter()

    @child.get("/x")
    def route_x() -> str:  # pragma: no cover
        return "x"

    @child.get("/y")
    def route_y() -> str:  # pragma: no cover
        return "y"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up.
    assert client.get("/v1/x").status_code == 200
    assert client.get("/v1/y").status_code == 200

    # Direct mutation.
    child.routes.clear()

    # Both routes must be gone.
    assert client.get("/v1/x").status_code == 404, (
        "Cache not invalidated after .clear()"
    )
    assert client.get("/v1/y").status_code == 404, (
        "Cache not invalidated after .clear()"
    )


# ---------------------------------------------------------------------------
# Test: cache warm-up is not required – plain append before any request
# ---------------------------------------------------------------------------


def test_direct_append_before_any_request():
    """
    Direct .append() before any request has been made must also work
    (no pre-existing cache state should get in the way).
    """
    child = APIRouter()

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Append a route with no prior request (cache never warmed).
    new_route = _make_route("/fresh", "fresh")
    child.routes.append(new_route)

    resp = client.get("/v1/fresh")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test: nested inclusion – mutation on grandchild invalidates through two hops
# ---------------------------------------------------------------------------


def test_direct_append_to_grandchild_routes_invalidates_cache():
    """
    The cache of a grandchild (child-of-child) should also be invalidated when
    routes are directly appended, because _get_routes_version() aggregates
    version numbers recursively from all nested _IncludedRouters.
    """
    grandchild = APIRouter()

    @grandchild.get("/ping")
    def ping() -> str:  # pragma: no cover
        return "pong"

    child = APIRouter()
    child.include_router(grandchild, prefix="/gc")

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up cache all the way down.
    resp = client.get("/v1/gc/ping")
    assert resp.status_code == 200

    # Directly mutate the grandchild.
    new_route = _make_route("/pong", "ping")
    grandchild.routes.append(new_route)

    # The full chain of caches must be invalidated.
    resp2 = client.get("/v1/gc/pong")
    assert resp2.status_code == 200, (
        f"Expected 200 after direct append to grandchild, got {resp2.status_code}. "
        "Nested cache was not invalidated."
    )


# ---------------------------------------------------------------------------
# Test: routes list is a proper list subclass (isinstance check)
# ---------------------------------------------------------------------------


def test_routes_is_list_subclass():
    """
    router.routes must remain a list (subclass) so that existing code
    doing isinstance(router.routes, list) continues to work.
    """
    router = APIRouter()
    assert isinstance(router.routes, list), (
        "router.routes must be an instance of list for backward compatibility"
    )


# ---------------------------------------------------------------------------
# Test: __setitem__ invalidates the cache
# ---------------------------------------------------------------------------


def test_setitem_on_child_routes_invalidates_cache():
    """
    Replacing a route via index assignment (child.routes[0] = new_route) must
    also invalidate the cache.
    """
    child = APIRouter()

    @child.get("/old")
    def route_old() -> str:  # pragma: no cover
        return "old"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up.
    assert client.get("/v1/old").status_code == 200

    # Replace the route via index assignment.
    child.routes[0] = _make_route("/new", "new")

    # Old route is gone, new route is present.
    assert client.get("/v1/old").status_code == 404, (
        "__setitem__ did not invalidate cache"
    )
    assert client.get("/v1/new").status_code == 200, "__setitem__ new route not found"


# ---------------------------------------------------------------------------
# Test: __delitem__ invalidates the cache
# ---------------------------------------------------------------------------


def test_delitem_on_child_routes_invalidates_cache():
    """
    Deleting a route via del child.routes[idx] must invalidate the cache.
    """
    child = APIRouter()

    @child.get("/delete-me")
    def route_delete() -> str:  # pragma: no cover
        return "bye"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up.
    assert client.get("/v1/delete-me").status_code == 200

    del child.routes[0]

    assert client.get("/v1/delete-me").status_code == 404, (
        "__delitem__ did not invalidate cache"
    )


def test_routes_mutation_coverage() -> None:
    """
    Ensure all other list mutation methods correctly trigger invalidation
    to maintain 100% coverage on _RouteList.
    """
    child = APIRouter()

    def r1() -> str:
        return "1"

    def r2() -> str:
        return "2"

    r1.__name__ = "r1"
    r2.__name__ = "r2"

    route1 = APIRoute("/r1", endpoint=r1, methods=["GET"])
    route2 = APIRoute("/r2", endpoint=r2, methods=["GET"])
    route3 = APIRoute("/r3", endpoint=r2, methods=["GET"])

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # 1. insert
    child.routes.insert(0, route1)
    assert client.get("/v1/r1").status_code == 200

    # 2. extend
    child.routes.extend([route2])
    assert client.get("/v1/r2").status_code == 200

    # 3. pop
    popped = child.routes.pop()
    assert popped == route2
    assert client.get("/v1/r2").status_code == 404

    # 4. __iadd__
    child.routes += [route2]
    assert client.get("/v1/r2").status_code == 200

    # 5. reverse
    child.routes.reverse()
    assert client.get("/v1/r2").status_code == 200

    # 6. __imul__
    child.routes *= 2
    assert len(child.routes) == 4
    # It just reorders them, but it should bump version
    assert client.get("/v1/r1").status_code == 200

    # 6. sort
    child.routes.append(route3)
    # Sort by path
    child.routes.sort(key=lambda r: getattr(r, "path", ""))
    assert client.get("/v1/r3").status_code == 200


# ---------------------------------------------------------------------------
# Tests: direct reassignment (router.routes = [...])
# ---------------------------------------------------------------------------


def test_routes_reassignment_type_is_route_list():
    """
    After ``router.routes = [...]`` the attribute must still be a
    :class:`_RouteList`, not a plain :class:`list`.
    """
    from fastapi.routing import _RouteList

    router = APIRouter()

    @router.get("/a")
    def _a() -> str:  # pragma: no cover
        return "a"

    router.routes = []
    assert isinstance(router.routes, _RouteList), (
        "router.routes must remain a _RouteList after direct reassignment"
    )


def test_routes_reassignment_new_route_reachable_after_cache_warmup():
    """
    Regression for the scenario reported by @BitWeaverDev:

        warmup: 200
        after reassignment, new route reachable? expected 200, got: 404

    After the cache has been warmed by a first request, a direct reassignment
    of ``child.routes`` must make the new routes reachable.
    """
    child = APIRouter()

    @child.get("/old")
    def route_old() -> str:  # pragma: no cover
        return "old"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up the candidate-route cache.
    warmup = client.get("/v1/old")
    assert warmup.status_code == 200, f"warmup failed: {warmup.status_code}"

    # Directly reassign routes with a new route.
    new_route = _make_route("/new", "new")
    child.routes = [new_route]

    # New route must be reachable.
    resp_new = client.get("/v1/new")
    assert resp_new.status_code == 200, (
        f"after reassignment, new route reachable? expected 200, got: {resp_new.status_code}"
    )


def test_routes_reassignment_old_route_gone_after_cache_warmup():
    """
    Regression for the scenario reported by @BitWeaverDev:

        after reassignment, old route gone? expected 404, got: 200

    After direct reassignment of ``child.routes``, the old routes must no
    longer be reachable (stale ``_IncludedRouter`` caches must be cleared).
    """
    child = APIRouter()

    @child.get("/old")
    def route_old() -> str:  # pragma: no cover
        return "old"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up.
    assert client.get("/v1/old").status_code == 200

    # Directly reassign routes – old route is NOT included.
    new_route = _make_route("/new", "new")
    child.routes = [new_route]

    # Old route must be gone.
    resp_old = client.get("/v1/old")
    assert resp_old.status_code == 404, (
        f"after reassignment, old route gone? expected 404, got: {resp_old.status_code}"
    )


def test_routes_reassignment_is_still_route_list_after_warmup():
    """
    Regression for the scenario reported by @BitWeaverDev:

        routes is still _RouteList after reassignment? list

    ``router.routes`` must be an ``_RouteList`` instance even after a warm
    cache and a direct reassignment.
    """
    from fastapi.routing import _RouteList

    child = APIRouter()

    @child.get("/x")
    def route_x() -> str:  # pragma: no cover
        return "x"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up.
    assert client.get("/v1/x").status_code == 200

    # Reassign.
    child.routes = [_make_route("/y", "y")]

    assert isinstance(child.routes, _RouteList), (
        f"routes is still _RouteList after reassignment? {type(child.routes).__name__}"
    )


def test_routes_reassignment_nested_router():
    """
    Direct reassignment on a nested (included) router must correctly
    invalidate caches through the full inclusion chain.
    """
    child = APIRouter()

    @child.get("/before")
    def before() -> str:  # pragma: no cover
        return "before"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up through two hops.
    assert client.get("/v1/before").status_code == 200

    # Replace child routes entirely.
    child.routes = [_make_route("/after", "after")]

    assert client.get("/v1/after").status_code == 200, (
        "new route not reachable after reassignment through included router"
    )
    assert client.get("/v1/before").status_code == 404, (
        "old route still reachable after reassignment through included router"
    )


def test_routes_slice_assignment_invalidates_cache():
    """
    Slice assignment (``router.routes[:] = [...]``) is handled by
    ``_RouteList.__setitem__`` and must also invalidate the cache.
    """
    child = APIRouter()

    @child.get("/alpha")
    def alpha() -> str:  # pragma: no cover
        return "alpha"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up.
    assert client.get("/v1/alpha").status_code == 200

    # Slice-assign to replace all routes.
    child.routes[:] = [_make_route("/beta", "beta")]

    assert client.get("/v1/beta").status_code == 200, (
        "new route not reachable after slice assignment"
    )
    assert client.get("/v1/alpha").status_code == 404, (
        "old route still reachable after slice assignment"
    )


def test_routes_reassignment_empty_list():
    """
    Assigning an empty list to ``router.routes`` must clear all routes and
    leave ``router.routes`` as a :class:`_RouteList`.
    """
    from fastapi.routing import _RouteList

    child = APIRouter()

    @child.get("/existing")
    def existing() -> str:  # pragma: no cover
        return "existing"

    app = FastAPI()
    app.include_router(child, prefix="/v1")
    client = TestClient(app, raise_server_exceptions=True)

    # Warm up.
    assert client.get("/v1/existing").status_code == 200

    # Clear via reassignment.
    child.routes = []

    assert isinstance(child.routes, _RouteList)
    assert len(child.routes) == 0
    assert client.get("/v1/existing").status_code == 404, (
        "route still reachable after routes reassigned to []"
    )
