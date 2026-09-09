"""Phase B tests: same-origin static UI serving (Option A) -- offline.

Covers the ``_mount_ui`` helper on fresh FastAPI instances: opt-in via
``SERVE_UI=1``, ``UI_DIST_DIR`` override, index.html served at ``/`` with
``html=True``, API-route precedence over the mount, missing-dist no-op, and
default-off (no env -> no mount). The module app keeps its import-time
behavior; these tests never touch it.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from convex_optimization.app import _mount_ui


def _fresh_app_with_api() -> FastAPI:
    """A minimal stand-in app with one API route registered BEFORE the mount."""
    a = FastAPI()

    @a.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return a


def _make_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><head><title>workbench</title></head>"
        "<body><div id=workbench></div></body></html>",
        encoding="utf-8",
    )
    (assets / "index-abc.js").write_text("console.log(1);", encoding="utf-8")
    return dist


def test_serves_index_html_at_root(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path)
    a = _fresh_app_with_api()
    os.environ["SERVE_UI"] = "1"
    os.environ["UI_DIST_DIR"] = str(dist)
    try:
        _mount_ui(a)
        client = TestClient(a)
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "workbench" in r.text
        # asset paths resolve under the mount
        r2 = client.get("/assets/index-abc.js")
        assert r2.status_code == 200
        assert "javascript" in r2.headers["content-type"]
        # unknown paths 404 honestly (no SPA fallback)
        assert client.get("/no-such-page").status_code == 404
    finally:
        del os.environ["SERVE_UI"]
        del os.environ["UI_DIST_DIR"]


def test_api_routes_keep_precedence_over_mount(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path)
    a = _fresh_app_with_api()
    os.environ["SERVE_UI"] = "1"
    os.environ["UI_DIST_DIR"] = str(dist)
    try:
        _mount_ui(a)
        client = TestClient(a)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
    finally:
        del os.environ["SERVE_UI"]
        del os.environ["UI_DIST_DIR"]


def test_no_mount_without_serve_ui(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path)
    a = _fresh_app_with_api()
    os.environ.pop("SERVE_UI", None)
    os.environ["UI_DIST_DIR"] = str(dist)
    try:
        _mount_ui(a)
        client = TestClient(a)
        assert client.get("/").status_code == 404
    finally:
        os.environ.pop("UI_DIST_DIR", None)


def test_no_mount_when_dist_missing(tmp_path: Path) -> None:
    a = _fresh_app_with_api()
    os.environ["SERVE_UI"] = "1"
    os.environ["UI_DIST_DIR"] = str(tmp_path / "absent")
    try:
        _mount_ui(a)
        client = TestClient(a)
        assert client.get("/").status_code == 404
    finally:
        del os.environ["SERVE_UI"]
        os.environ.pop("UI_DIST_DIR", None)
