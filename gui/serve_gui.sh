#!/bin/bash
# Runs the GUI -- run this from the root of the repo

source venv/bin/activate

cd gui
# python3 -m http.server 7998
# uvicorn main:app --reload --port 7998 --host 0.0.0.0
python3 main.py 
