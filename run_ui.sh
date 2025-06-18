#!/usr/bin/env bash

PYTHONPATH=$(pwd) uv run --env-file .env streamlit run ui/app.py
