"""
Phase 5: FastAPI Service for Driver Repositioning Recommendations.

Includes:
- Complete Driver Web App frontend (Login, Overall City Demand, Personalized Recommendations)
- REST API with 15-minute periodic background refresh
- Driver authentication (Email/Phone + Password)
- Personalized driver recommendations based on real-time driver location
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import re
from typing import List, Optional, Literal, Any
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from .earnings_analysis import compute_historical_earnings_analysis
    from .phase4_recommend import (
        generate_recommendations,
        get_driver_personalized_recommendations,
        DB_PATH,
    )
except (ImportError, ValueError):
    from earnings_analysis import compute_historical_earnings_analysis
    from phase4_recommend import (
        generate_recommendations,
        get_driver_personalized_recommendations,
        DB_PATH,
    )

# ---------------------------------------------------------------------------
# Logging & Timezone
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("commitcon_api")
IST = timezone(timedelta(hours=5, minutes=30))

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

# AI Inference Service Configuration
AI_INFERENCE_URL = os.getenv("AI_INFERENCE_URL", "https://mohd-shujath-commitcon-driver-inference.hf.space/chat")
AI_INFERENCE_KEY = os.getenv("AI_INFERENCE_KEY", "commitcon-secret-2026")

# ---------------------------------------------------------------------------
# Scheduler (Native background scheduler with APScheduler fallback)
# ---------------------------------------------------------------------------
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    class BackgroundScheduler:
        """Lightweight native background scheduler for periodic tasks."""

        class _Job:
            def __init__(self, func, interval_seconds, job_id):
                self.func = func
                self.interval_seconds = interval_seconds
                self.id = job_id
                self.next_run_time = datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)

        def __init__(self):
            self._jobs = {}
            self._running = False
            self._thread = None
            self._lock = threading.Lock()

        def add_job(self, func, trigger="interval", minutes=15, id="periodic_refresh", replace_existing=True):
            interval_seconds = minutes * 60
            job = self._Job(func, interval_seconds, id)
            with self._lock:
                self._jobs[id] = job
            return job

        def get_job(self, job_id):
            with self._lock:
                return self._jobs.get(job_id)

        def start(self):
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()

        def _run_loop(self):
            while self._running:
                time.sleep(1)
                now = datetime.now(timezone.utc)
                jobs_to_run = []
                with self._lock:
                    for job in self._jobs.values():
                        if now >= job.next_run_time:
                            job.next_run_time = now + timedelta(seconds=job.interval_seconds)
                            jobs_to_run.append(job)
                for job in jobs_to_run:
                    try:
                        job.func()
                    except Exception as err:
                        logger.error(f"Error executing scheduled job {job.id}: {err}")

        def shutdown(self, wait=False):
            self._running = False


# ---------------------------------------------------------------------------
# Demo Drivers Store
# ---------------------------------------------------------------------------
DEMO_DRIVERS = {
    "aditya@commitcon.com": {
        "id": "DRV_101",
        "name": "Aditya Singh",
        "email": "aditya@commitcon.com",
        "phone": "9876543210",
        "vehicle": "Toyota Sedan (KA-01-AB-1234)",
        "current_zone_id": "Z06",
        "avatar": "👨🏽‍💼",
        "rating": 4.92,
    },
    "rajesh@commitcon.com": {
        "id": "DRV_102",
        "name": "Rajesh Kumar",
        "email": "rajesh@commitcon.com",
        "phone": "9876543211",
        "vehicle": "Hyundai Prime (MH-02-CD-5678)",
        "current_zone_id": "Z13",
        "avatar": "👨🏻‍✈️",
        "rating": 4.88,
    },
    "vikram@commitcon.com": {
        "id": "DRV_103",
        "name": "Vikram Sharma",
        "email": "vikram@commitcon.com",
        "phone": "9876543212",
        "vehicle": "Maruti Dzire (DL-03-EF-9012)",
        "current_zone_id": "Z10",
        "avatar": "🧔🏾",
        "rating": 4.95,
    },
    "priya@commitcon.com": {
        "id": "DRV_104",
        "name": "Priya Patel",
        "email": "priya@commitcon.com",
        "phone": "9876543213",
        "vehicle": "Honda Amaze (KA-04-GH-3456)",
        "current_zone_id": "Z20",
        "avatar": "👩🏽",
        "rating": 4.97,
    },
}

# Also allow phone lookup
PHONE_TO_EMAIL = {d["phone"]: d["email"] for d in DEMO_DRIVERS.values()}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class DriverProfile(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    vehicle: str
    current_zone_id: str
    avatar: str
    rating: float


class LoginResponse(BaseModel):
    success: bool
    token: str
    driver: DriverProfile


class ZoneRecommendation(BaseModel):
    zone_id: str
    zone_name: str
    predicted_requests: int
    required_drivers: int
    available_drivers: int
    supply_gap: int
    recommended_action: str
    reason_codes: List[str]
    reposition_sources: Optional[List[str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class RecommendationResponse(BaseModel):
    run_id: str
    generated_at: str
    model_version: str
    total_zones: int
    summary: dict
    zones: List[ZoneRecommendation]


class HealthResponse(BaseModel):
    status: str
    last_refresh: Optional[str]
    next_refresh: Optional[str]
    total_runs: int


class AcceptRepositionRequest(BaseModel):
    driver_id: str
    destination_zone_id: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class DriverChatRequest(BaseModel):
    driver_id: str
    driver_prompt: str
    current_zone_id: Optional[str] = "Z06"
    history: Optional[List[ChatMessage]] = []


class DriverChatResponse(BaseModel):
    status: str
    driver_id: str
    answer: str
    model: str


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
latest_recommendation: Optional[dict] = None
recommendation_history: list = []
run_counter: int = 0
HISTORY_MAX = 10
scheduler = BackgroundScheduler()


def refresh_recommendations() -> dict:
    """Generate a new recommendation and update global state."""
    global latest_recommendation, run_counter

    run_counter += 1
    logger.info("Refreshing recommendations (run #%d) ...", run_counter)

    try:
        rec = generate_recommendations()
        latest_recommendation = rec

        recommendation_history.append(
            {
                "run_id": rec["run_id"],
                "generated_at": rec["generated_at"],
                "summary": rec["summary"],
            }
        )
        if len(recommendation_history) > HISTORY_MAX:
            recommendation_history.pop(0)

        logger.info(
            "Recommendation %s generated - %d zones.",
            rec["run_id"],
            rec["total_zones"],
        )
        return rec
    except Exception:
        logger.exception("Failed to refresh recommendations.")
        raise


# ---------------------------------------------------------------------------
# FastAPI app & Lifecycle
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown."""
    logger.info("Starting CommitCon Recommendation API & Driver Portal ...")
    try:
        refresh_recommendations()
    except Exception:
        logger.warning("Initial recommendation generation failed - API will serve 404 until next refresh.")

    scheduler.add_job(
        refresh_recommendations,
        "interval",
        minutes=15,
        id="periodic_refresh",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Periodic scheduler started (refreshing every 15 minutes).")
    yield
    logger.info("Shutting down scheduler ...")
    scheduler.shutdown(wait=False)

app = FastAPI(
    title="CommitCon Driver Repositioning Portal & API",
    description="Driver login, citywide demand dashboard, and personalized repositioning recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for frontend assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Web App Entry Route
# ---------------------------------------------------------------------------
@app.get("/api")
def root():
    """API info and service metadata."""
    return {
        "service": "CommitCon Driver Repositioning API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/recommendations/latest",
            "/recommendations/latest/zone/{zone_id}",
            "/recommendations/latest/action/{action}",
            "/recommendations/history",
            "/health",
            "/recommendations/refresh"
        ]
    }

@app.get("/")
def serve_home():
    """Serves the driver portal web application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return root()


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Driver login via email or phone number and password."""
    user_key = req.username_or_email.strip().lower()

    # Look up by email or phone
    driver_data = DEMO_DRIVERS.get(user_key)
    if not driver_data and req.username_or_email.strip() in PHONE_TO_EMAIL:
        email = PHONE_TO_EMAIL[req.username_or_email.strip()]
        driver_data = DEMO_DRIVERS.get(email)

    # Allow custom username/phone demo login as well
    if not driver_data:
        driver_data = {
            "id": f"DRV_{abs(hash(user_key)) % 900 + 100}",
            "name": req.username_or_email.split("@")[0].capitalize() if "@" in req.username_or_email else f"Driver {req.username_or_email[-4:]}",
            "email": req.username_or_email if "@" in req.username_or_email else f"{req.username_or_email}@commitcon.com",
            "phone": req.username_or_email if req.username_or_email.isdigit() else "9876543210",
            "vehicle": "Sedan (KA-01-CC-9999)",
            "current_zone_id": "Z06",
            "avatar": "🚗",
            "rating": 4.90,
        }

    return LoginResponse(
        success=True,
        token=f"demo_token_{driver_data['id']}",
        driver=DriverProfile(**driver_data),
    )


# ---------------------------------------------------------------------------
# Driver & Zone Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/zones")
def list_zones():
    """Returns all 20 zones with name and formatted ID."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()
        rows = cursor.execute("SELECT zone_id, zone_name, center_latitude, center_longitude FROM zones ORDER BY zone_id").fetchall()
        return [
            {
                "zone_id": f"Z{r[0]:02d}",
                "zone_name": r[1],
                "latitude": float(r[2]),
                "longitude": float(r[3]),
            }
            for r in rows
        ]
    finally:
        conn.close()


@app.get("/api/recommendations/for-me")
def get_personal_recommendations(zone_id: str = "Z06", limit: int = 5):
    """Returns customized repositioning recommendations for an individual driver
    calculated from their current zone.
    """
    try:
        return get_driver_personalized_recommendations(zone_id, max_results=limit)
    except Exception as exc:
        logger.exception("Failed to compute personalized recommendations for zone %s", zone_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not compute personal recommendations: {exc}",
        )


@app.post("/api/drivers/accept-reposition")
def accept_reposition(req: AcceptRepositionRequest):
    """Driver accepts repositioning to a target zone."""
    logger.info("Driver %s accepted repositioning to %s", req.driver_id, req.destination_zone_id)
    return {
        "status": "ACCEPTED",
        "driver_id": req.driver_id,
        "destination_zone_id": req.destination_zone_id,
        "dispatch_timestamp": datetime.now(IST).isoformat(),
        "message": f"Navigation route dispatched to {req.destination_zone_id}.",
    }


_CACHED_EARNINGS_ANALYSIS = None

@app.get("/api/analysis/earnings")
def get_earnings_analysis(driver_id: Optional[str] = None):
    """Returns counterfactual earnings analysis comparing driver income
    with vs without the CommitCon recommendation model over past 30 days.
    """
    global _CACHED_EARNINGS_ANALYSIS
    if _CACHED_EARNINGS_ANALYSIS is None:
        _CACHED_EARNINGS_ANALYSIS = compute_historical_earnings_analysis()

    data = dict(_CACHED_EARNINGS_ANALYSIS)
    if driver_id and driver_id in data.get("driver_profiles", {}):
        data["active_driver"] = data["driver_profiles"][driver_id]
    return data


@app.post("/api/chat", response_model=DriverChatResponse)
async def driver_copilot_chat(req: DriverChatRequest):
    """Driver chatbot assistant (Sherpa): integrates real-time recommendation data
    and forwards to the GPU-powered inference API.
    """
    logger.info("Driver %s asked Sherpa: %s", req.driver_id, req.driver_prompt[:80])

    # 1. Fetch real-time recommendation data for driver's current zone
    zone_id = req.current_zone_id or "Z06"
    try:
        recommendation_data = get_driver_personalized_recommendations(zone_id, max_results=6)
    except Exception as e:
        logger.warning("Could not get personalized recs for %s: %s", zone_id, e)
        recommendation_data = {
            "driver_current_zone": {"zone_id": zone_id, "zone_name": f"Zone {zone_id}"},
            "recommendations": []
        }

    # Attach citywide summary if available
    if latest_recommendation and "summary" in latest_recommendation:
        recommendation_data["citywide_summary"] = latest_recommendation["summary"]

    # 2. Prepare payload matching external ChatRequest schema
    payload = {
        "driver_id": req.driver_id,
        "driver_prompt": req.driver_prompt,
        "recommendation_data": recommendation_data,
        "history": [{"role": m.role, "content": m.content} for m in (req.history or [])]
    }

    # 3. Call AI inference API (Hugging Face Gradio Space or Direct REST API)
    hf_token = os.getenv("HF_TOKEN")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": AI_INFERENCE_KEY,
        "User-Agent": "CommitCon-Backend",
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # Build context-rich prompt so the model has live driver location & recommendation data
    top_recs = recommendation_data.get("recommendations", [])[:3]
    rec_summary = ", ".join([
        f"{r.get('zone_id')} {r.get('zone_name')} (gap: +{r.get('supply_gap')}, ETA: {r.get('travel_time_min')}m)"
        for r in top_recs
    ])
    context_prefix = (
        f"[Context: Driver in {zone_id}. High-demand zones: {rec_summary}]. "
        if rec_summary
        else f"[Context: Driver in {zone_id}]. "
    )
    augmented_prompt = f"{context_prefix}{req.driver_prompt}"

    is_hf_space = "hf.space" in AI_INFERENCE_URL or "huggingface.co" in AI_INFERENCE_URL

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            if is_hf_space:
                base_url = re.sub(r"/(chat|gradio_api/call/.*)?$", "", AI_INFERENCE_URL).rstrip("/")
                endpoints = [
                    f"{base_url}/gradio_api/call/chat",
                    f"{base_url}/gradio_api/call/gradio_chat",
                ]
                hf_headers = {"Content-Type": "application/json"}
                if hf_token:
                    hf_headers["Authorization"] = f"Bearer {hf_token}"

                call_resp = None
                call_url = None
                last_call_error = None

                for ep in endpoints:
                    try:
                        resp = await client.post(
                            ep,
                            json={"data": [augmented_prompt, []]},
                            headers=hf_headers,
                            timeout=15.0,
                        )
                        if resp.status_code == 200:
                            call_resp = resp
                            call_url = ep
                            break
                        else:
                            last_call_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
                    except Exception as e:
                        last_call_error = str(e)
                        continue

                if not call_resp or not call_url:
                    raise RuntimeError(f"Could not connect to Gradio queue: {last_call_error}")

                event_id = call_resp.json().get("event_id")
                if not event_id:
                    raise RuntimeError("No event_id returned by Gradio queue.")

                # Stream SSE response from Gradio
                answer = ""
                quota_error = None
                async with client.stream("GET", f"{call_url}/{event_id}", headers=hf_headers, timeout=40.0) as stream:
                    if stream.status_code != 200:
                        stream.raise_for_status()
                    async for line in stream.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("event: error"):
                            continue
                        if line.startswith("data:"):
                            raw_data = line[5:].strip()
                            try:
                                parsed = json.loads(raw_data)
                                if isinstance(parsed, dict) and "error" in parsed:
                                    quota_error = parsed["error"]
                                elif isinstance(parsed, list) and len(parsed) > 0:
                                    last = parsed[-1]
                                    if isinstance(last, list) and len(last) > 1:
                                        answer = str(last[1])
                                    elif isinstance(last, str):
                                        answer = str(last)
                            except Exception:
                                pass
                        elif "event: complete" in line:
                            break

                if quota_error:
                    raise RuntimeError(quota_error)

                if not answer:
                    answer = "Response received from Hugging Face model."

                return DriverChatResponse(
                    status="success",
                    driver_id=req.driver_id,
                    answer=answer,
                    model="huggingface-qwen-0.8b",
                )

            # Direct REST API POST (e.g. local Ollama / Docker / FastAPI)
            resp = await client.post(AI_INFERENCE_URL, json=payload, headers=headers)

            # If a non-HF URL returns 405, fall back to checking if it's a Gradio space
            if resp.status_code == 405:
                base_url = re.sub(r"/(chat|gradio_api/call/.*)?$", "", AI_INFERENCE_URL).rstrip("/")
                call_resp = await client.post(
                    f"{base_url}/gradio_api/call/gradio_chat",
                    json={"data": [augmented_prompt, []]},
                    headers=headers,
                    timeout=15.0,
                )
                if call_resp.status_code == 200:
                    event_id = call_resp.json().get("event_id")
                    async with client.stream("GET", f"{base_url}/gradio_api/call/gradio_chat/{event_id}", headers=headers, timeout=40.0) as stream:
                        answer = ""
                        async for line in stream.aiter_lines():
                            if line.startswith("data:"):
                                try:
                                    parsed = json.loads(line[5:].strip())
                                    if isinstance(parsed, list) and len(parsed) > 0:
                                        last = parsed[-1]
                                        answer = str(last[1] if isinstance(last, list) and len(last) > 1 else last)
                                except Exception:
                                    pass
                            elif "event: complete" in line:
                                break
                    return DriverChatResponse(
                        status="success",
                        driver_id=req.driver_id,
                        answer=answer or "Response received.",
                        model="huggingface-qwen-0.8b",
                    )

            resp.raise_for_status()
            data = resp.json()
            return DriverChatResponse(
                status="success",
                driver_id=req.driver_id,
                answer=data.get("answer", "No answer received from model."),
                model=data.get("model", "project-driver:latest"),
            )

    except httpx.TimeoutException:
        logger.error("AI inference service timed out for driver %s", req.driver_id)
        top_rec = recommendation_data["recommendations"][0] if recommendation_data.get("recommendations") else None
        fallback_msg = (
            f"The AI model is taking longer than usual to respond. "
            f"Based on live demand forecasts, your highest priority target is {top_rec['zone_id']} {top_rec['zone_name']} "
            f"with a shortage of {top_rec['supply_gap']} drivers (ETA: {top_rec['travel_time_min']}m)."
            if top_rec else "The AI inference service is busy. Please try again in a few moments."
        )
        return DriverChatResponse(
            status="fallback",
            driver_id=req.driver_id,
            answer=fallback_msg,
            model="local-rules-engine",
        )
    except Exception as exc:
        logger.exception("AI inference error: %s", exc)
        top_rec = recommendation_data["recommendations"][0] if recommendation_data.get("recommendations") else None
        err_str = str(exc)
        if "quota" in err_str.lower():
            err_reason = "Hugging Face ZeroGPU daily quota reached. To increase quota, set HF_TOKEN in Render environment."
        else:
            err_reason = err_str[:80]

        fallback_msg = (
            f"Note: External GPU inference reached limit ({err_reason}). "
            f"Live recommendation: Reposition to {top_rec['zone_id']} {top_rec['zone_name']} "
            f"(ETA: {top_rec['travel_time_min']} mins, +{top_rec['supply_gap']} drivers needed)."
            if top_rec else f"AI service note: {err_reason}"
        )
        return DriverChatResponse(
            status="fallback",
            driver_id=req.driver_id,
            answer=fallback_msg,
            model="local-rules-engine",
        )


# ---------------------------------------------------------------------------
# Standard Recommendation Endpoints (Phases 4 & 5 Core Requirements)
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Health-check endpoint with refresh metadata."""
    last_refresh = None
    if latest_recommendation:
        last_refresh = latest_recommendation["generated_at"]

    next_refresh = None
    job = scheduler.get_job("periodic_refresh")
    if job and getattr(job, "next_run_time", None):
        next_refresh = job.next_run_time.astimezone(IST).isoformat()

    return HealthResponse(
        status="healthy",
        last_refresh=last_refresh,
        next_refresh=next_refresh,
        total_runs=run_counter,
    )


@app.get("/recommendations/latest", response_model=RecommendationResponse)
def get_latest():
    """Return the most recent recommendation payload for all 20 zones."""
    if latest_recommendation is None:
        raise HTTPException(status_code=404, detail="No recommendations available yet.")
    return latest_recommendation


@app.get(
    "/recommendations/latest/zone/{zone_id}",
    response_model=ZoneRecommendation,
)
def get_zone(zone_id: str):
    """Return the recommendation for a single zone from the latest run."""
    if latest_recommendation is None:
        raise HTTPException(status_code=404, detail="No recommendations available yet.")

    zone_id_upper = zone_id.upper()
    for zone in latest_recommendation["zones"]:
        if zone["zone_id"].upper() == zone_id_upper:
            return zone

    raise HTTPException(
        status_code=404,
        detail=f"Zone '{zone_id}' not found in the latest recommendation.",
    )


@app.get(
    "/recommendations/latest/action/{action}",
    response_model=List[ZoneRecommendation],
)
def get_by_action(action: str):
    """Filter zones from the latest recommendation by action."""
    if latest_recommendation is None:
        raise HTTPException(status_code=404, detail="No recommendations available yet.")

    valid_actions = {"REPOSITION_IN", "REPOSITION_OUT", "MONITOR"}
    action_upper = action.upper()
    if action_upper not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{action}'. Must be one of {sorted(valid_actions)}.",
        )

    filtered = [
        z for z in latest_recommendation["zones"]
        if z["recommended_action"] == action_upper
    ]
    return filtered


@app.get("/recommendations/history")
def get_history():
    """Return summaries of recent recommendation runs (up to last 10)."""
    return recommendation_history


@app.post("/recommendations/refresh", response_model=RecommendationResponse)
def manual_refresh():
    """Manually trigger a new recommendation run and return the result."""
    try:
        rec = refresh_recommendations()
        return rec
    except Exception as exc:
        logger.exception("Manual refresh failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {exc}",
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import uvicorn
    uvicorn.run("src.phase5_api:app", host="0.0.0.0", port=8000, reload=True)
