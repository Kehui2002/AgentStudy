from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
from typing import Any

from origin_fit.contracts import WorkerSubmission

from .service import OriginWorker, WorkerError


def create_app(worker: OriginWorker, *, bearer_token: str) -> Any:
    """Create the optional FastAPI `/v1` transport adapter."""
    if not bearer_token:
        raise ValueError("Worker Bearer Token must not be empty")
    try:
        from fastapi import (  # type: ignore[import-not-found]
            Depends,
            FastAPI,
            Header,
            HTTPException,
            status,
        )
        from fastapi.responses import Response  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on installation extra
        raise RuntimeError(
            "Install the 'origin-worker' extra to serve the Worker API."
        ) from error

    @asynccontextmanager
    async def lifespan(app: Any):  # type: ignore[no-untyped-def]
        worker.cleanup_expired_workspaces()
        startup_task = asyncio.create_task(worker.run_queued())
        app.state.queue_tasks.add(startup_task)
        startup_task.add_done_callback(task_finished)
        try:
            yield
        finally:
            tasks = tuple(app.state.queue_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def task_finished(task: asyncio.Task[None]) -> None:
        app.state.queue_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    app = FastAPI(title="Origin Worker", version="1.0", lifespan=lifespan)
    app.state.queue_tasks = set()

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
        submission: WorkerSubmission,
        _: None = Depends(authenticate),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, object]:
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
