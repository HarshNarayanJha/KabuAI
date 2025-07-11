#!/usr/bin/env bash

cd ui-streamlit
uv run --env-file .env.ui streamlit run app.py
