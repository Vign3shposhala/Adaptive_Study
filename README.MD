# Adaptive Study Coach - OpenEnv RL Environment

## Overview
A real-world RL environment simulating student learning optimization.

## Features
- Focus, energy, stress modeling
- Action-based adaptive learning
- Reward-driven improvement

## API
- POST /reset
- POST /step
- GET /state

## Run

### Local
uvicorn server.app:app --reload

### Inference
python inference.py

## Tasks
- learning-efficiency
- stress-control
- focus-optimization

## Compliance
- OpenEnv compatible
- FastAPI server
- Docker ready
- Scores normalized [0,1]