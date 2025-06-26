# Real-Time Web Object Detection

> **Domain:** Manufacturing

## Overview

Operations depend on cameras but lack real-time, in-browser detection to surface what matters. Manual monitoring is slow, inconsistent, and hard to scale across sites. Offline video review makes incident response lag behind floor activity. Leaders need lightweight ways to turn any device camera into a smart sensor without heavy infrastructure. Without this capability, teams spend hours combing footage, miss defects or safety events, and delay corrective actions impacting throughput and customer commitments. This project delivers a browser-based application detecting objects live, drawing bounding boxes instantly, and logging cropped evidence into reports enabling supervisors to act fast while keeping data within company systems.

## Approach

- Ran discovery with operations to define object classes, camera angles, alert thresholds, reporting needs; mapped user journeys for operators and supervisors
- Prepared training dataset; applied augmentation and image preprocessing; trained custom object-detection model tuned for commodity device cameras and on-device latency
- Built Python Flask backend with SQLite for lightweight persistence of detections, cropped images, timestamps, and user actions
- Implemented secure HTML5 camera access to stream video, perform inference, draw bounding boxes in-browser, capture crops for audit-ready reports
- Validated with held-out test set and field trials; measured precision/recall and false positives
- Delivered in sprints with weekly demos; added role-based access, basic analytics, web UI for reviewing, filtering, exporting detections

## Skills & Technologies

- Python
- Flask
- SQLite
- OpenCV
- Object Detection
- Model Training
- HTML5 Camera APIs
- REST API Design
