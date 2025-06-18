#!/usr/bin/env bash

PYTHONPATH=$(pwd) uv run --env-file .env python ventureai/main.py
