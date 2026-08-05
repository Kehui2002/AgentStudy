from __future__ import annotations

import asyncio
from collections.abc import Collection
from contextlib import asynccontextmanager
import hmac
from typing import Any

from pydantic import ValidationError

from origin_fit.contracts import WorkerSubmission

from .service import OriginWorker, WorkerError


def create_app(
    worker: OriginWorker,
    *,
    bearer_token: str,
    cleanup_interval: float = 60 * 60,
    max_upload_bytes: int | None = None,
    allowed_client_hosts: Collection[str] | None = None,
) -> Any:
    """Create the optional FastAPI `/v1` transport adapter."""
    if not bearer_token:
        raise ValueError("Worker Bearer Token must not be empty")
    if cleanup_interval <= 0:
        raise ValueError("cleanup_interval must be positive")
    if max_upload_bytes is None:
        max_upload_bytes = worker.max_submission_bytes
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")
    try:
        from fastapi import (  # type: ignore[import-not-found]
            Depends,
            FastAPI,
            Header,
            HTTPException,
            Request,
            status,
        )
        from fastapi.responses import (  # type: ignore[import-not-found]
            JSONResponse,
            Response,
        )
    except ImportError as error:  # pragma: no cover - depends on installation extra
        raise RuntimeError(
            "Install the 'origin-worker' extra to serve the Worker API."
        ) from error
    # FastAPI resolves postponed annotations against module globals.
    globals()["Request"] = Request

    @asynccontextmanager
    async def lifespan(app: Any):  # type: ignore[no-untyped-def]
        worker.cleanup_expired_workspaces()
        startup_task = asyncio.create_task(worker.run_queued())
        app.state.queue_tasks.add(startup_task)
        startup_task.add_done_callback(task_finished)
        cleanup_task = asyncio.create_task(cleanup_forever())
        app.state.queue_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(task_finished)
        try:
            yield
        finally:
            tasks = tuple(app.state.queue_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def cleanup_forever() -> None:
        while True:
            await asyncio.sleep(cleanup_interval)
            worker.cleanup_expired_workspaces()

    def task_finished(task: asyncio.Task[None]) -> None:
        app.state.queue_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    app = FastAPI(title="Origin Worker", version="1.0", lifespan=lifespan)
    app.state.queue_tasks = set()
    allowed_hosts = (
        None if allowed_client_hosts is None else frozenset(allowed_client_hosts)
    )

    @app.middleware("http")
    async def restrict_client(request: Request, call_next: Any) -> Any:
        is_v1_request = request.url.path == "/v1" or request.url.path.startswith(
            "/v1/"
        )
        if is_v1_request and allowed_hosts is not None:
            client_host = request.client.host if request.client is not None else None
            if client_host not in allowed_hosts:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": {
                            "code": "client_not_allowed",
                            "message": "Client source is not allowed.",
                        }
                    },
                )
        return await call_next(request)

    async def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {bearer_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "unauthorized", "message": "Authentication failed."},
            )

    def worker_error(error: WorkerError) -> HTTPException:
        response_status = (
            status.HTTP_404_NOT_FOUND
            if error.code == "not_found"
            else status.HTTP_409_CONFLICT
            if error.code == "idempotency_conflict"
            else status.HTTP_400_BAD_REQUEST
        )
        return HTTPException(
            status_code=response_status,
            detail={"code": error.code, "message": error.message},
        )

    @app.get("/v1/health", dependencies=[Depends(authenticate)])
    async def health() -> dict[str, str]:
        return worker.health()

    @app.get("/v1/capabilities", dependencies=[Depends(authenticate)])
    async def capabilities() -> dict[str, object]:
        return worker.capabilities().model_dump(mode="json")

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def submit(
        request: Request,
        _: None = Depends(authenticate),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, object]:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = max_upload_bytes + 1
            if declared_length > max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail={
                        "code": "upload_too_large",
                        "message": "Worker submission exceeds the upload limit.",
                    },
                )
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail={
                        "code": "upload_too_large",
                        "message": "Worker submission exceeds the upload limit.",
                    },
                )
            body.extend(chunk)
        try:
            submission = WorkerSubmission.model_validate_json(body, strict=True)
        except ValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "invalid_submission",
                    "message": "Worker submission is invalid.",
                },
            ) from error
        try:
            job = worker.submit(submission, idempotency_key or "")
        except WorkerError as error:
            raise worker_error(error) from error
        task = asyncio.create_task(worker.run_queued())
        app.state.queue_tasks.add(task)
        task.add_done_callback(task_finished)
        return job.model_dump(mode="json")

    @app.get("/v1/jobs/{worker_job_id}")
    async def get_job(
        worker_job_id: str, _: None = Depends(authenticate)
    ) -> dict[str, object]:
        try:
            return worker.get_job(worker_job_id).model_dump(mode="json")
        except WorkerError as error:
            raise worker_error(error) from error

    @app.post("/v1/jobs/{worker_job_id}/cancel")
    async def cancel(
        worker_job_id: str, _: None = Depends(authenticate)
    ) -> dict[str, object]:
        try:
            return worker.cancel(worker_job_id).model_dump(mode="json")
        except WorkerError as error:
            raise worker_error(error) from error

    @app.get("/v1/jobs/{worker_job_id}/bundle")
    async def bundle(
        worker_job_id: str, _: None = Depends(authenticate)
    ) -> Response:
        try:
            content = worker.get_bundle(worker_job_id)
        except WorkerError as error:
            raise worker_error(error) from error
        return Response(content, media_type="application/zip")

    return app
