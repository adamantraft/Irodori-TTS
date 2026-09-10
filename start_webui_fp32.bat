@echo off
cd /d %~dp0
SET TTS_PRECISION=fp32
start http://localhost:8501
uv run --no-sync streamlit run webui.py --server.port 8501
