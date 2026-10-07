# UI

A clean Streamlit frontend for the PSI02 Temporal Video Intelligence project.

## 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run

```bash
streamlit run app.py
```

The interface starts with an empty state. No dataset or fake events are bundled.

## 3. Logo

The app looks for:

```text
assets/logo.png
```

Your provided project logo can be copied there.

You can also point to a different logo file:

```text
KAJU_LOGO_PATH=path/to/logo.png
```

If no logo exists, the UI uses a small `KK` fallback mark so the header does not break.

## 4. Connect Person 1

Person 1 owns detection and tracking.

Set:

```text
KAJU_VISION_MODULE=person1_vision
KAJU_VISION_FUNCTION=analyze_video
```

Expected call:

```python
tracks = analyze_video(video_path)
```

Return any Python object containing the tracking output.

## 5. Connect Person 2

Person 2 owns event extraction and temporal memory.

Set:

```text
KAJU_EVENT_MODULE=person2_events
KAJU_EVENT_FUNCTION=extract_events
```

Expected call:

```python
events = extract_events(tracks)
```

Each event should ideally look like:

```python
{
    "timestamp": 42.3,
    "event": "entered",
    "entity_id": "3",
    "entity_type": "person",
    "zone": "Zone A",
    "confidence": 0.94
}
```

The UI also accepts the simpler forms used in the project, such as:

```python
{
    "id": 3,
    "type": "person",
    "event": "entered",
    "timestamp": 42.3
}
```

## 6. Connect Person 3

Person 3 owns question understanding, temporal reasoning, and answer generation.

Set:

```text
KAJU_QA_MODULE=person3_qa
KAJU_QA_FUNCTION=answer_question
```

Preferred call:

```python
result = answer_question(
    question,
    events,
    video_path,
)
```

A good result is:

```python
{
    "answer": "Person #7 entered Zone B at 02:17.",
    "timestamp": 137,
    "source_events": [
        {"event": "truck_arrival", "timestamp": 129},
        {"event": "person_enter_zone", "entity_id": "7", "timestamp": 137}
    ]
}
```

A simple string is also accepted:

```python
return "Person #7 entered Zone B at 02:17."
```

## 7. Architecture

```text
Uploaded Video
      |
      v
Person 1: Detection + Tracking
      |
      v
Tracking Output
      |
      v
Person 2: Event Extraction + Temporal Memory
      |
      v
Structured Events
      |
      v
Person 3: Question Understanding + Temporal Reasoning
      |
      v
Answer + Timestamp + Evidence
      |
      v
Kaju Katli Dashboard
```

## 8. Important integration rule

Do not make the UI depend on a particular YOLO model, tracker, dataset, or LLM.

Only keep the contracts above stable. This means Person 1, Person 2, and Person 3 can change their internal implementations without forcing a rewrite of the frontend.

## 9. Where to customize

The main functions to edit are:

- `run_person1()`
- `run_person2()`
- `run_person3()`

Everything else is presentation and session management.
