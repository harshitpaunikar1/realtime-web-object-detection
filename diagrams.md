# Real-Time Web Object Detection Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## Browser inference pipeline

```mermaid
flowchart TD
    N1["Step 1\nRan discovery with operations to define object classes, camera angles, alert thres"]
    N2["Step 2\nPrepared training dataset; applied augmentation and image preprocessing; trained c"]
    N1 --> N2
    N3["Step 3\nBuilt Python Flask backend with SQLite for lightweight persistence of detections, "]
    N2 --> N3
    N4["Step 4\nImplemented secure HTML5 camera access to stream video, perform inference, draw bo"]
    N3 --> N4
    N5["Step 5\nValidated with held-out test set and field trials; measured precision/recall and f"]
    N4 --> N5
```

## Detection → log → report flow

```mermaid
flowchart LR
    N1["Inputs\nMedical PDFs, guidelines, or evidence documents"]
    N2["Decision Layer\nDetection → log → report flow"]
    N1 --> N2
    N3["User Surface\nAPI-facing integration surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
