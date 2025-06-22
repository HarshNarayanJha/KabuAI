#!/usr/bin/env bash

cd kabuai
uv run uvicorn main:app --reload --env-file ../.env
