# start.sh
# Use Render's PORT or fallback to 10000
uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --reload
