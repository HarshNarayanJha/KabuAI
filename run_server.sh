#!/usr/bin/env bash

cd ventureai
uvicorn main:app --reload --env-file ../.env
