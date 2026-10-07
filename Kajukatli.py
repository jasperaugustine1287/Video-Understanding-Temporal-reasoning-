from __future__ import annotations

import importlib
import base64
import csv
import io
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import streamlit as st


# ============================================================
# KAJU KATLI — Temporal Video Intelligence UI
# ============================================================
# This frontend intentionally contains NO preset dataset.
#
# Integration points:
#   Person 1 -> vision/tracking
#   Person 2 -> event extraction/temporal memory
#   Person 3 -> question answering/reasoning
#
# The UI works immediately with empty state and can be connected
# to the team's real Python modules without changing the layout.
# ============================================================


APP_NAME = "Kaju Katli"
APP_TAGLINE = "Temporal Video Intelligence"
MAX_EVENTS_IN_STREAM = 500


# ---------------------------
# Data contracts
# ---------------------------

@dataclass
class Event:
    """Canonical event format passed around the UI.

    Keep these fields stable when integrating Person 1/2/3 modules.
    Extra fields are preserved in the payload dictionary.
    """
    timestamp: float
    event: str
    entity_id: str = ""
    entity_type: str = ""
    duration: float | None = None
    zone: str = ""
    related_to: str = ""
    confidence: float | None = None
    payload: dict[str, Any] | None = None

    @classmethod
    def from_any(cls, item: Any) -> "Event":
        if isinstance(item, cls):
            return item

        if not isinstance(item, dict):
            raise TypeError(f"Unsupported event type: {type(item)!r}")

        timestamp = item.get(
            "timestamp",
            item.get("time", item.get("ts", item.get("start_time", 0.0))),
        )

        return cls(
            timestamp=float(timestamp or 0.0),
            event=str(item.get("event", item.get("type", "event"))),
            entity_id=str(item.get("entity_id", item.get("id", ""))),
            entity_type=str(item.get("entity_type", item.get("class", ""))),
            duration=_to_float_or_none(item.get("duration")),
            zone=str(item.get("zone", "")),
            related_to=str(item.get("related_to", item.get("relationship", ""))),
            confidence=_to_float_or_none(item.get("confidence")),
            payload=item,
        )


def _to_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------
# Integration adapters
# ---------------------------

def _load_callable(module_name: str, callable_name: str) -> Callable | None:
    """Load an integration function only when configured.

    Example:
      export KAJU_VISION_MODULE=person1_vision
      export KAJU_VISION_FUNCTION=analyze_video
    """
    if not module_name or not callable_name:
        return None

    try:
        module = importlib.import_module(module_name)
        function = getattr(module, callable_name, None)
        return function if callable(function) else None
    except Exception:
        return None


def get_vision_adapter() -> Callable | None:
    return _load_callable(
        os.getenv("KAJU_VISION_MODULE", ""),
        os.getenv("KAJU_VISION_FUNCTION", "analyze_video"),
    )


def get_event_adapter() -> Callable | None:
    return _load_callable(
        os.getenv("KAJU_EVENT_MODULE", ""),
        os.getenv("KAJU_EVENT_FUNCTION", "extract_events"),
    )


def get_qa_adapter() -> Callable | None:
    return _load_callable(
        os.getenv("KAJU_QA_MODULE", ""),
        os.getenv("KAJU_QA_FUNCTION", "answer_question"),
    )


def run_person1(video_path: str) -> Any:
    adapter = get_vision_adapter()
    if adapter is None:
        raise RuntimeError(
            "Person 1 module is not connected. "
            "Configure KAJU_VISION_MODULE and KAJU_VISION_FUNCTION."
        )
    return adapter(video_path)


def run_person2(tracks: Any) -> list[Event]:
    adapter = get_event_adapter()
    if adapter is None:
        raise RuntimeError(
            "Person 2 module is not connected. "
            "Configure KAJU_EVENT_MODULE and KAJU_EVENT_FUNCTION."
        )

    raw_events = adapter(tracks)
    return [Event.from_any(item) for item in (raw_events or [])]


def run_person3(question: str, events: list[Event], video_path: str | None = None) -> dict[str, Any]:
    adapter = get_qa_adapter()
    if adapter is None:
        raise RuntimeError(
            "Person 3 module is not connected. "
            "Configure KAJU_QA_MODULE and KAJU_QA_FUNCTION."
        )

    # Flexible contract: Person 3 may accept either 3 args or 2 args.
    try:
        result = adapter(question, [asdict(e) for e in events], video_path)
    except TypeError:
        result = adapter(question, [asdict(e) for e in events])

    if isinstance(result, str):
        return {"answer": result}

    return dict(result or {})


# ---------------------------
# App state
# ---------------------------

def initialize_state() -> None:
    defaults = {
        "events": [],
        "tracks": None,
        "video_path": None,
        "video_name": None,
        "analysis_status": "Ready",
        "analysis_message": "Upload a video to begin.",
        "selected_event_timestamp": None,
        "timeline_event_filter": [],
        "timeline_person_filter": [],
        "answer": None,
        "answer_timestamp": None,
        "answer_source_events": [],
        "last_question": "",
        "pipeline_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------
# Formatting helpers
# ---------------------------

def format_time(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--"

    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def event_label(event: Event) -> str:
    parts = [event.event.replace("_", " ").title()]
    if event.entity_id:
        parts.append(f"#{event.entity_id}")
    if event.zone:
        parts.append(f"· {event.zone}")
    return " ".join(parts)


def normalize_events(events: Iterable[Any]) -> list[Event]:
    normalized = [Event.from_any(e) for e in events]
    normalized.sort(key=lambda e: e.timestamp)
    return normalized[:MAX_EVENTS_IN_STREAM]


# ---------------------------
# CSS / UI
# ---------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --kk-red: #d71920;
            --kk-black: #000000;
            --kk-panel: #111111;
            --kk-panel-2: #171717;
            --kk-border: #2b2b2b;
            --kk-white: #ffffff;
            --kk-muted: #b8b8b8;
        }

        /* Keep only the native sidebar control in Streamlit's header.
           The old version hid the entire header, which also removed the
           arrow used to reopen the left panel. */
        [data-testid="stHeader"] {
            background: #000000 !important;
            border-bottom: 0 !important;
            height: 2.75rem !important;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button {
            color: #ffffff !important;
            background: transparent !important;
            border: 0 !important;
        }

        footer {
            visibility: hidden !important;
            height: 0 !important;
        }

        html, body,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        .stApp {
            background: var(--kk-black) !important;
            color: var(--kk-white) !important;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 4.0rem !important;
            padding-bottom: 3.5rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
        }

        /* Make regular Streamlit-generated text white on the black canvas. */
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] p {
            color: var(--kk-white) !important;
        }

        section[data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            transform: translateX(0) !important;
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            bottom: 0 !important;
            width: 320px !important;
            min-width: 320px !important;
            z-index: 100000 !important;
            background: #080808 !important;
            border-right: 1px solid var(--kk-border) !important;
            overflow-y: auto !important;
        }

        /* Keep the main workspace clear of the fixed left panel. */
        [data-testid="stAppViewContainer"] {
            padding-left: 320px !important;
        }

        /* The native collapse control is intentionally hidden because
           the sidebar itself is kept permanently visible. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        section[data-testid="stSidebar"] * {
            color: var(--kk-white) !important;
        }

        section[data-testid="stSidebar"] .stFileUploader section {
            background: #0f0f0f !important;
            border-color: #3a3a3a !important;
        }

        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stDownloadButton > button {
            background: #ffffff !important;
            color: #111111 !important;
            border: 0 !important;
            border-radius: 10px;
            min-height: 42px;
        }

        /* Keep the secondary sidebar button text visible on its white surface. */
        section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]),
        section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) *,
        section[data-testid="stSidebar"] .stDownloadButton > button,
        section[data-testid="stSidebar"] .stDownloadButton > button * {
            color: #111111 !important;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"],
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
            background: var(--kk-red) !important;
            color: #ffffff !important;
        }

        .kk-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.45rem 0 1.45rem 0;
            border-bottom: 1px solid var(--kk-border);
            margin-bottom: 1.7rem;
        }

        .kk-brand {
            display: flex;
            align-items: center;
            gap: 1.0rem;
            min-width: 0;
            padding-left: 0.1rem;
        }

        .kk-logo {
            width: 66px;
            height: 66px;
            min-width: 66px;
            min-height: 66px;
            border-radius: 50%;
            object-fit: cover;
            object-position: center;
            flex: 0 0 66px;
            border: 1px solid #333333;
            display: block;
            transform: scale(1.08);
        }

        .kk-logo-fallback {
            width: 54px;
            height: 54px;
            border-radius: 50%;
            background: var(--kk-red);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 800;
            font-size: 1.0rem;
            letter-spacing: 0.04em;
        }

        .kk-brand-name {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--kk-white);
            line-height: 1.1;
        }

        .kk-brand-sub {
            font-size: 0.82rem;
            color: var(--kk-muted);
            margin-top: 0.2rem;
        }

        .kk-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.8rem;
            border: 1px solid #3a3a3a;
            border-radius: 999px;
            color: var(--kk-white);
            background: var(--kk-panel);
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .kk-status-dot {
            width: 8px;
            height: 8px;
            background: var(--kk-red);
            border-radius: 50%;
        }

        .kk-title {
            font-size: 2rem;
            font-weight: 800;
            color: var(--kk-white);
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }

        .kk-subtitle {
            color: #c3c3c3;
            margin-bottom: 1.4rem;
        }

        .kk-card {
            border: 1px solid var(--kk-border);
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            background: var(--kk-panel);
            box-shadow: 0 8px 28px rgba(0,0,0,0.25);
            height: 100%;
        }

        .kk-card-soft {
            background: var(--kk-panel-2);
        }

        .kk-card-title {
            color: var(--kk-white);
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 0.7rem;
        }

        .kk-mini {
            color: #a6a6a6;
            font-size: 0.78rem;
        }

        .kk-value {
            color: var(--kk-white);
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }

        .kk-answer {
            border-left: 4px solid var(--kk-red);
            padding: 0.95rem 1rem;
            background: #0c0c0c;
            border-radius: 8px;
            color: var(--kk-white);
            line-height: 1.55;
            font-size: 1rem;
        }

        .kk-event {
            border: 1px solid var(--kk-border);
            border-radius: 12px;
            padding: 0.72rem 0.85rem;
            margin-bottom: 0.6rem;
            background: #0d0d0d;
        }

        .kk-event-time {
            color: #ff454c;
            font-weight: 800;
            font-size: 0.78rem;
        }

        .kk-event-name {
            color: var(--kk-white);
            font-weight: 750;
            margin-top: 0.15rem;
        }

        .kk-event-meta {
            color: #a9a9a9;
            font-size: 0.76rem;
            margin-top: 0.18rem;
        }

        .kk-empty {
            min-height: 190px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            color: #aaaaaa;
            border: 1px dashed #3a3a3a;
            border-radius: 14px;
            background: #0d0d0d;
            padding: 1.5rem;
        }

        .kk-empty strong {
            color: #ffffff;
            display: block;
            margin-bottom: 0.3rem;
        }

        div.stButton > button {
            border-radius: 10px;
            border: 1px solid #444444;
            background: #111111;
            color: #ffffff !important;
            min-height: 42px;
            font-weight: 700;
        }

        div.stButton > button:hover {
            border-color: var(--kk-red);
            color: #ffffff !important;
        }

        div.stButton > button[kind="primary"] {
            background: var(--kk-red) !important;
            border-color: var(--kk-red) !important;
            color: #ffffff !important;
        }

        .stTextInput input,
        .stTextArea textarea {
            border: 1px solid #444444 !important;
            border-radius: 10px !important;
            color: #ffffff !important;
            background: #101010 !important;
            caret-color: #ffffff !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #858585 !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: var(--kk-red) !important;
            box-shadow: 0 0 0 1px var(--kk-red) !important;
        }

        .stFileUploader section {
            border: 1px dashed #414141 !important;
            border-radius: 12px;
            background: #0d0d0d !important;
        }

        .stFileUploader section * {
            color: #ffffff !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--kk-border);
            border-radius: 12px;
            overflow: hidden;
        }

        .kk-sidebar-note {
            background: #111111;
            border: 1px solid #2b2b2b;
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            font-size: 0.78rem;
            color: #d0d0d0;
            line-height: 1.4;
        }

        .kk-arch {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.65rem;
        }

        .kk-arch-step {
            border: 1px solid #303030;
            border-radius: 11px;
            padding: 0.75rem;
            text-align: center;
            background: #101010;
        }

        .kk-arch-step b {
            display: block;
            color: #ffffff;
            font-size: 0.8rem;
        }

        .kk-arch-step span {
            display: block;
            color: #999999;
            font-size: 0.72rem;
            margin-top: 0.15rem;
        }

        .stAlert {
            background: #161616 !important;
            color: #ffffff !important;
            border-color: #333333 !important;
        }

        [data-testid="stExpander"] {
            background: #0f0f0f !important;
            border: 1px solid #2c2c2c !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background: #0f0f0f !important;
            color: #ffffff !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1.2rem;
                padding-right: 1.2rem;
            }

            .kk-arch {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_logo() -> None:
    logo_path = Path(os.getenv("KAJU_LOGO_PATH", "assets/logo.png"))

    if logo_path.exists():
        st.image(str(logo_path), width=46)
        return

    st.markdown('<div class="kk-logo-fallback">KK</div>', unsafe_allow_html=True)


# ---------------------------
# Sidebar
# ---------------------------

def render_sidebar() -> Any:
    with st.sidebar:
        st.markdown("## Kaju Katli")
        st.caption("Temporal Video Intelligence")

        st.markdown("### 1 · Video")
        uploaded = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            label_visibility="collapsed",
        )

        st.markdown("### 2 · Analysis")
        analyze = st.button(
            "Run full pipeline",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

        clear = st.button("Clear session", use_container_width=True)

        st.markdown("### 3 · Filters")
        entity_filter = st.multiselect(
            "Entity type",
            options=["person", "object", "machine"],
            default=[],
        )

        event_filter = st.text_input(
            "Event contains",
            placeholder="e.g. entered",
        )

        st.markdown("---")
        st.markdown(
            '<div class="kk-sidebar-note">'
            "<b>Integration ready</b><br>"
            "Connect Person 1, 2 and 3 through the adapter functions in app.py. "
            "The interface does not require a preset dataset."
            "</div>",
            unsafe_allow_html=True,
        )

    if clear:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    return uploaded, analyze, entity_filter, event_filter


# ---------------------------
# Pipeline
# ---------------------------

def save_uploaded_video(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.flush()
    temp.close()
    return temp.name


def run_pipeline(uploaded_file: Any) -> None:
    st.session_state.pipeline_error = None
    st.session_state.answer = None
    st.session_state.selected_event_timestamp = None
    st.session_state.timeline_event_filter = []
    st.session_state.timeline_person_filter = []

    video_path = save_uploaded_video(uploaded_file)
    st.session_state.video_path = video_path
    st.session_state.video_name = uploaded_file.name
    st.session_state.analysis_status = "Analyzing"

    try:
        tracks = run_person1(video_path)
        st.session_state.tracks = tracks

        events = run_person2(tracks)
        st.session_state.events = normalize_events(events)
        if st.session_state.events:
            st.session_state.timeline_time_filter = (
                float(min(event.timestamp for event in st.session_state.events)),
                float(max(event.timestamp for event in st.session_state.events)),
            )

        st.session_state.analysis_status = "Ready"
        st.session_state.analysis_message = (
            f"{len(st.session_state.events)} events extracted successfully."
        )
    except Exception as exc:
        st.session_state.analysis_status = "Needs integration"
        st.session_state.pipeline_error = str(exc)
        st.session_state.analysis_message = (
            "The UI is working, but the analysis adapters are not connected yet."
        )


def run_question(question: str) -> None:
    if not question.strip():
        return

    st.session_state.last_question = question

    try:
        result = run_person3(
            question=question,
            events=st.session_state.events,
            video_path=st.session_state.video_path,
        )
        st.session_state.answer = str(result.get("answer", result.get("text", "")))
        st.session_state.answer_timestamp = _to_float_or_none(
            result.get("timestamp", result.get("time"))
        )
        st.session_state.answer_source_events = result.get(
            "source_events",
            result.get("evidence", []),
        )
    except Exception as exc:
        st.session_state.answer = None
        st.session_state.pipeline_error = str(exc)


# ---------------------------
# Main content sections
# ---------------------------

def render_topbar() -> None:
    logo_path = Path(os.getenv("KAJU_LOGO_PATH", "assets/logo.png"))

    if logo_path.exists():
        encoded = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        logo_html = f'<img class="kk-logo" src="data:image/png;base64,{encoded}" alt="Kaju Katli logo">'
    else:
        logo_html = '<div class="kk-logo-fallback">KK</div>'

    st.markdown(
        f"""
        <div class="kk-topbar">
            <div class="kk-brand">
                {logo_html}
                <div>
                    <div class="kk-brand-name">Kaju Katli</div>
                    <div class="kk-brand-sub">Temporal Video Intelligence</div>
                </div>
            </div>
            <div class="kk-status">
                <span class="kk-status-dot"></span>
                Analysis workspace
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    events = st.session_state.events
    people = len({e.entity_id for e in events if e.entity_type.lower() == "person" and e.entity_id})
    objects = len({e.entity_id for e in events if e.entity_type.lower() in {"object", "machine"} and e.entity_id})

    st.markdown('<div class="kk-title">Ask the video.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="kk-subtitle">Upload one video, build a temporal event memory, and query what happened, who did it, when, and how events relate.</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            f'<div class="kk-card"><div class="kk-mini">VIDEO</div><div class="kk-value">{st.session_state.video_name or "Not loaded"}</div></div>',
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f'<div class="kk-card"><div class="kk-mini">EVENTS</div><div class="kk-value">{len(events)}</div></div>',
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f'<div class="kk-card"><div class="kk-mini">PERSON ENTITIES</div><div class="kk-value">{people}</div></div>',
            unsafe_allow_html=True,
        )

    with d:
        st.markdown(
            f'<div class="kk-card"><div class="kk-mini">OTHER ENTITIES</div><div class="kk-value">{objects}</div></div>',
            unsafe_allow_html=True,
        )


def render_video_and_stream(uploaded: Any, entity_filter: list[str], event_filter: str) -> None:
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.markdown(
            '<div class="kk-card"><div class="kk-card-title">Video workspace</div>',
            unsafe_allow_html=True,
        )

        start_time = st.session_state.selected_event_timestamp
        if uploaded is not None:
            st.video(uploaded, start_time=int(start_time or 0))
        elif st.session_state.video_path and Path(st.session_state.video_path).exists():
            st.video(st.session_state.video_path, start_time=int(start_time or 0))
        else:
            st.markdown(
                '<div class="kk-empty"><strong>No video loaded</strong>Use the upload control in the left panel to add a video for analysis.</div>',
                unsafe_allow_html=True,
            )
        if start_time is not None:
            st.caption(f"Timeline seek point: {float(start_time):.2f}s — press play to review.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="kk-card"><div class="kk-card-title">Live event stream</div>',
            unsafe_allow_html=True,
        )

        events = st.session_state.events

        if entity_filter:
            events = [e for e in events if e.entity_type.lower() in entity_filter]

        if event_filter.strip():
            needle = event_filter.strip().lower()
            events = [e for e in events if needle in e.event.lower()]

        if not events:
            st.markdown(
                '<div class="kk-empty"><strong>Waiting for events</strong>The stream will populate after Person 1 and Person 2 are connected.</div>',
                unsafe_allow_html=True,
            )
        else:
            stream = events[-25:]
            for event in reversed(stream):
                meta = []
                if event.entity_type:
                    meta.append(event.entity_type.title())
                if event.zone:
                    meta.append(event.zone)
                if event.confidence is not None:
                    meta.append(f"{event.confidence:.0%} confidence")

                st.markdown(
                    f"""
                    <div class="kk-event">
                        <div class="kk-event-time">{format_time(event.timestamp)}</div>
                        <div class="kk-event-name">{event_label(event)}</div>
                        <div class="kk-event-meta">{' · '.join(meta) if meta else 'Temporal event'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


def render_timeline() -> None:
    st.markdown("### Timeline")

    events = st.session_state.events
    if not events:
        st.markdown(
            '<div class="kk-empty"><strong>Timeline is empty</strong>Extracted events will appear here in chronological order.</div>',
            unsafe_allow_html=True,
        )
        return

    # Compact overview of event intervals and the raw observations they group.
    by_type: dict[str, list[Event]] = {}
    for event in events:
        by_type.setdefault(event.event, []).append(event)
    raw_observation_count = sum(
        int((event.payload or {}).get("grouped_count", 1))
        for event in events
    )
    summary_cols = st.columns(3)
    summary_cols[0].metric("Event intervals", len(events))
    summary_cols[1].metric("Event types", len(by_type))
    summary_cols[2].metric("Observations summarized", raw_observation_count)
    breakdown = []
    for event_type, items in sorted(by_type.items()):
        duration = sum(event.duration or 0.0 for event in items)
        breakdown.append({
            "Event": event_type.replace("_", " ").title(),
            "Intervals": len(items),
            "Approx. grouped time": f"{duration:.1f}s" if duration else "—",
        })
    with st.expander("Event counts and approximate spans"):
        st.caption("Grouped durations estimate the span between sampled event timestamps.")
        st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)

    st.markdown("#### Filter timeline")
    filter_cols = st.columns([1, 1, 1.4])
    event_types = sorted({event.event for event in events})
    with filter_cols[0]:
        selected_types = st.multiselect(
            "Event type",
            options=event_types,
            default=st.session_state.timeline_event_filter,
            format_func=lambda value: value.replace("_", " ").title(),
            key="timeline_event_filter",
        )

    person_ids = set()
    for event in events:
        payload = event.payload or {}
        if payload.get("global_id") is not None:
            person_ids.add(str(payload["global_id"]))
        person_ids.update(str(person_id) for person_id in payload.get("people", []))
        if event.entity_type.lower() == "person" and event.entity_id:
            person_ids.add(event.entity_id)
        elif event.entity_type.lower() == "people" and event.entity_id:
            person_ids.update(part.strip() for part in event.entity_id.split("&"))

    with filter_cols[1]:
        selected_people = st.multiselect(
            "Person",
            options=sorted(person_ids),
            default=st.session_state.timeline_person_filter,
            key="timeline_person_filter",
        )

    min_time = min(event.timestamp for event in events)
    max_time = max(event.timestamp for event in events)
    slider_max = max_time if max_time > min_time else min_time + 1.0
    previous_range = st.session_state.get("timeline_time_filter")
    default_range = (
        tuple(previous_range)
        if previous_range and len(previous_range) == 2
        else (float(min_time), float(slider_max))
    )
    with filter_cols[2]:
        selected_range = st.slider(
            "Time range (seconds)",
            min_value=float(min_time),
            max_value=float(slider_max),
            value=default_range,
            step=0.5,
            key="timeline_time_filter",
        )

    filtered_events = []
    for event in events:
        payload = event.payload or {}
        linked_people = set()
        if payload.get("global_id") is not None:
            linked_people.add(str(payload["global_id"]))
        linked_people.update(str(person_id) for person_id in payload.get("people", []))
        if event.entity_id:
            linked_people.update(part.strip() for part in event.entity_id.split("&"))

        if selected_types and event.event not in selected_types:
            continue
        if selected_people and not linked_people.intersection(selected_people):
            continue
        event_end = float(payload.get("end_timestamp", event.timestamp))
        if event_end < selected_range[0] or event.timestamp > selected_range[1]:
            continue
        filtered_events.append(event)

    if not filtered_events:
        st.info("No events match these filters.")
        return

    rows = []
    for idx, event in enumerate(filtered_events):
        rows.append(
            {
                "#": idx + 1,
                "Time": f"{format_time(event.timestamp)} ({event.timestamp:.2f}s)",
                "Through": (
                    f"{float((event.payload or {}).get('end_timestamp', event.timestamp)):.2f}s"
                    if (event.payload or {}).get("end_timestamp") is not None
                    else ""
                ),
                "Event": event.event.replace("_", " ").title(),
                "Entity": f"{event.entity_type.title()} #{event.entity_id}" if event.entity_id else event.entity_type.title(),
                "Zone": event.zone,
                "Duration": f"{event.duration:.2f}s" if event.duration is not None else "",
            }
        )

    selection = st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=270,
        on_select="rerun",
        selection_mode="single-row",
        key="timeline_table",
    )

    selected_rows = selection.selection.rows
    if selected_rows:
        selected_event = filtered_events[selected_rows[0]]
        if st.session_state.selected_event_timestamp != selected_event.timestamp:
            st.session_state.selected_event_timestamp = selected_event.timestamp
            st.rerun()

    # Export the currently filtered, grouped timeline.
    export_rows = []
    export_events = []
    for event in filtered_events:
        payload = dict(event.payload or {})
        payload.setdefault("timestamp", event.timestamp)
        payload.setdefault("type", event.event)
        if event.entity_id:
            payload.setdefault("entity_id", event.entity_id)
        if event.entity_type:
            payload.setdefault("entity_type", event.entity_type)
        if event.duration is not None:
            payload.setdefault("duration", event.duration)
        export_events.append(payload)
        export_rows.append({
            "timestamp_seconds": event.timestamp,
            "end_timestamp_seconds": (event.payload or {}).get("end_timestamp", event.timestamp),
            "event_type": event.event,
            "entity_type": event.entity_type,
            "entity_id": event.entity_id,
            "duration_seconds": event.duration,
            "description": (event.payload or {}).get("description", ""),
        })

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(export_rows[0].keys()))
    writer.writeheader()
    writer.writerows(export_rows)
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download filtered timeline (CSV)",
        data=csv_buffer.getvalue(),
        file_name="temporal_timeline.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download filtered timeline (JSON)",
        data=json.dumps(export_events, indent=2),
        file_name="temporal_timeline.json",
        mime="application/json",
        use_container_width=True,
    )


def render_question_panel() -> None:
    st.markdown("### Ask the video")

    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "Natural-language question",
            value=st.session_state.last_question,
            placeholder="e.g. What happened before the alarm?",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Ask",
            type="primary",
            use_container_width=True,
        )

    if submitted and question.strip():
        run_question(question.strip())

    if st.session_state.answer:
        timestamp_text = (
            f"Timestamp: {format_time(st.session_state.answer_timestamp)}"
            if st.session_state.answer_timestamp is not None
            else "Timestamp: provided by reasoning layer"
        )

        st.markdown(
            f"""
            <div class="kk-card kk-card-soft">
                <div class="kk-card-title">Answer</div>
                <div class="kk-answer">{st.session_state.answer}</div>
                <div class="kk-mini" style="margin-top:0.7rem;">{timestamp_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sources = st.session_state.answer_source_events
        if sources:
            with st.expander("Evidence used"):
                st.json(sources)

    else:
        st.markdown(
            '<div class="kk-card kk-card-soft"><div class="kk-card-title">Answer</div><div class="kk-empty" style="min-height:130px;"><strong>No question answered yet</strong>Ask about before / after / during relationships, counts, durations, or entity relationships.</div></div>',
            unsafe_allow_html=True,
        )


def render_architecture() -> None:
    st.markdown("### Integration architecture")

    st.markdown(
        """
        <div class="kk-arch">
            <div class="kk-arch-step"><b>1 · Vision</b><span>YOLO + tracking</span></div>
            <div class="kk-arch-step"><b>2 · Events</b><span>Temporal memory</span></div>
            <div class="kk-arch-step"><b>3 · Reasoning</b><span>Question → evidence</span></div>
            <div class="kk-arch-step"><b>4 · UI</b><span>Answer + timestamp</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_debug() -> None:
    with st.expander("Developer integration console"):
        st.caption("These diagnostics are for connecting the three team modules. They are not required for the final demo.")

        c1, c2, c3 = st.columns(3)

        with c1:
            vision = get_vision_adapter()
            st.write("Person 1 · Vision:", "Connected" if vision else "Not connected")

        with c2:
            event = get_event_adapter()
            st.write("Person 2 · Events:", "Connected" if event else "Not connected")

        with c3:
            qa = get_qa_adapter()
            st.write("Person 3 · QA:", "Connected" if qa else "Not connected")

        if st.session_state.pipeline_error:
            st.error(st.session_state.pipeline_error)

        if st.session_state.tracks is not None:
            st.caption("Raw tracking output preview")
            try:
                st.json(st.session_state.tracks)
            except Exception:
                st.write(st.session_state.tracks)

        if st.session_state.events:
            st.download_button(
                "Export event memory as JSON",
                data=json.dumps([asdict(e) for e in st.session_state.events], indent=2),
                file_name="kaju_katli_events.json",
                mime="application/json",
                use_container_width=False,
            )


# ---------------------------
# Entry point
# ---------------------------

def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_state()
    inject_css()
    render_topbar()

    uploaded, analyze, entity_filter, event_filter = render_sidebar()

    if analyze and uploaded is not None:
        with st.spinner("Running Vision → Events → Temporal Memory…"):
            run_pipeline(uploaded)
        st.rerun()

    if uploaded is not None and st.session_state.video_name is None:
        st.session_state.video_name = uploaded.name

    render_overview()
    st.markdown("")

    render_video_and_stream(uploaded, entity_filter, event_filter)

    st.markdown("")
    render_timeline()

    st.markdown("")
    render_question_panel()

    st.markdown("")
    render_architecture()

    st.markdown("")
    render_debug()

    st.caption(
        "Kaju Katli · Frontend shell for PSI02 Temporal Video Intelligence · "
        "No dataset is bundled with this interface."
    )


if __name__ == "__main__":
    main()
