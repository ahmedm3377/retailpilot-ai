"""FastAPI app: a thin, typed HTTP layer over DemoService (which wraps the shared AgentRuntime).

Bind to 127.0.0.1 only (see app.api.serve). The React dev server proxies /api
and /health here, so no CORS is configured. Tokens arrive in X-Demo-Session.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import psycopg
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agents.model_client import ModelClient, ModelUnavailableError
from app.agents.schemas import CustomerMessage, json_schema
from app.agents.settings import AgentSettings
from app.api.schemas import (
    ActionRequest, AuditEventDTO, CaseOptionDTO, CreatedSessionDTO, CreateSession, EndRequest, ReadyDTO, RequestStatusDTO,
    SessionDTO, TurnRequest, TurnResponseDTO,
)
from app.api.service import ApiProblem, DemoService, Resources
from app.db.connection import database_url
from app.guardrails.service import GuardrailService
from app.observability import telemetry as tel

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "candidate_assets" / "public_cases.json"
TOKEN_HEADER = "X-Demo-Session"


class ResourceLoader:
    """Loads the shared model client and guardrails once, in the background, at startup."""

    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self.resources: Resources | None = None
        self.error: str | None = None
        self.model_info: dict[str, Any] | None = None
        self._checked_at = 0.0

    def start(self) -> None:
        threading.Thread(target=self._load, name="resource-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            guards = GuardrailService(self.settings)
            model = ModelClient(self.settings)
            self.model_info = model.ready()
            model.chat_json("warmup", [{"role": "user", "content": "Reply with the message: ready"}],
                            json_schema(CustomerMessage))
            self.resources = Resources(model, guards, self.settings)
        except Exception as error:  # noqa: BLE001 - reported by /health/ready
            self.error = f"{type(error).__name__}: {str(error)[:200]}"

    def __call__(self) -> Resources | None:
        return self.resources

    def model_server(self) -> dict[str, Any]:
        """Live model server check, cached for 10 s."""
        if self.resources is not None and time.monotonic() - self._checked_at > 10:
            try:
                self.model_info = self.resources.model.ready()
                self._checked_at = time.monotonic()
            except ModelUnavailableError:
                self.model_info = None
        info = self.model_info or {}
        return {"ok": bool(info), "version": info.get("engine_version"), "model": info.get("model"),
                "digest": (info.get("digest") or "")[:12] or None}


def _error(status: int, code: str, message: str) -> JSONResponse:
    body = {"error_code": code, "message": message, "correlation_id": uuid.uuid4().hex[:12]}
    return JSONResponse(status_code=status, content=body)


def create_app(*, service_factory: Callable[[], DemoService] | None = None, loader: Any = None) -> FastAPI:
    """service_factory/loader are injectable for tests; production builds both from the environment."""
    settings = AgentSettings.from_env()
    loader = loader or ResourceLoader(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tel.init()  # no-op unless RETAIL_OTEL_ENABLED
        if isinstance(loader, ResourceLoader):
            loader.start()
        app.state.service = service_factory() if service_factory else DemoService(database_url(), FIXTURES, loader)
        yield
        app.state.service.close()
        tel.shutdown()

    app = FastAPI(title="Local Returns Desk API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ApiProblem)
    async def _problem(_: Request, exc: ApiProblem) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e.get("loc", [])[1:]) for e in exc.errors()})
        return _error(400, "validation", f"Invalid request fields: {', '.join(fields) or 'body'}.")

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        if request.method == "POST":
            return _error(503, "outcome_unverified", "The action could not be completed reliably. "
                                                     "Check the request status before trying again.")
        return _error(500, "internal_error", "Unexpected server error.")

    def svc(request: Request) -> DemoService:
        return request.app.state.service

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", response_model=ReadyDTO)
    def ready(request: Request) -> ReadyDTO:
        try:
            with psycopg.connect(svc(request).url, connect_timeout=3) as conn:
                pg = {"ok": True, "version": conn.execute("SHOW server_version").fetchone()[0]}
        except psycopg.Error:
            pg = {"ok": False, "version": None}
        loaded = loader() is not None
        guards = loader().guards if loaded else None
        unavailable = sorted(guards.unavailable) if guards else []
        model_server = (loader.model_server() if hasattr(loader, "model_server")
                        else {"ok": loaded, "version": None, "model": None, "digest": None})
        error = getattr(loader, "error", None)
        state = ("degraded" if error or not pg["ok"] or unavailable or (loaded and not model_server["ok"])
                 else "ready" if loaded else "loading")
        return ReadyDTO(state=state, postgres=pg, model_server=model_server,
                        guardrails={"loaded": loaded and not unavailable, "unavailable": unavailable})

    @app.get("/api/v1/demo/cases", response_model=list[CaseOptionDTO])
    def list_cases(request: Request) -> list[CaseOptionDTO]:
        return svc(request).case_options()

    @app.post("/api/v1/demo/sessions", response_model=CreatedSessionDTO)
    def create_session(body: CreateSession, request: Request) -> CreatedSessionDTO:
        token, session = svc(request).create_session(body.customer_id, body.scenario_id, body.uploaded_case)
        return CreatedSessionDTO(token=token, session=session)

    @app.get("/api/v1/demo/session", response_model=SessionDTO)
    def get_session(request: Request, x_demo_session: str | None = Header(None)) -> SessionDTO:
        service = svc(request)
        return service.session_view(service.authenticate(x_demo_session))

    def respond(result: tuple[int, Any], span: Any = None) -> JSONResponse:
        status, body = result
        trace_id = tel.trace_id_hex(span) if span is not None else None
        tel.set_attributes(span, {"http.response.status_code": status})
        return JSONResponse(status_code=status, content=body,
                            headers={"X-Trace-Id": trace_id} if trace_id else None)

    def route_span(route: str) -> Any:
        # Route TEMPLATE only (never the literal proposal ID) on a server span.
        return tel.span(f"POST {route}", {"http.request.method": "POST", "http.route": route}, server=True)

    @app.post("/api/v1/turns", response_model=TurnResponseDTO)
    def turn(body: TurnRequest, request: Request, x_demo_session: str | None = Header(None)) -> JSONResponse:
        service = svc(request)
        with route_span("/api/v1/turns") as span:
            return respond(service.send_message(service.authenticate(x_demo_session), body.message,
                                                body.client_action_id), span)

    @app.post("/api/v1/proposals/{proposal_id}/confirm", response_model=TurnResponseDTO)
    def confirm(proposal_id: str, body: ActionRequest, request: Request,
                x_demo_session: str | None = Header(None)) -> JSONResponse:
        service = svc(request)
        with route_span("/api/v1/proposals/{proposal_id}/confirm") as span:
            return respond(service.confirm(service.authenticate(x_demo_session), proposal_id[:64],
                                           body.client_action_id), span)

    @app.post("/api/v1/proposals/{proposal_id}/cancel", response_model=RequestStatusDTO)
    def cancel(proposal_id: str, body: ActionRequest, request: Request,
               x_demo_session: str | None = Header(None)) -> JSONResponse:
        service = svc(request)
        with route_span("/api/v1/proposals/{proposal_id}/cancel") as span:
            return respond(service.cancel(service.authenticate(x_demo_session), proposal_id[:64],
                                          body.client_action_id), span)

    @app.get("/api/v1/request-status", response_model=RequestStatusDTO)
    def request_status(request: Request, x_demo_session: str | None = Header(None)) -> RequestStatusDTO:
        service = svc(request)
        return service.request_status(service.authenticate(x_demo_session))

    @app.get("/api/v1/audit", response_model=list[AuditEventDTO])
    def audit(request: Request, x_demo_session: str | None = Header(None)) -> list[AuditEventDTO]:
        service = svc(request)
        return service.audit(service.authenticate(x_demo_session))

    @app.post("/api/v1/demo/sessions/{session_id}/end")
    def end_session(session_id: str, body: EndRequest, request: Request,
                    x_demo_session: str | None = Header(None)) -> dict[str, bool]:
        service = svc(request)
        service.end_session(service.authenticate(x_demo_session), session_id)
        return {"ended": True}

    return app


app = create_app() if os.environ.get("RETAIL_API_NO_DEFAULT_APP") != "1" else None
