@echo off
cd /d %~dp0
start http://localhost:8501
uv run streamlit run webui.py --server.port 8501
