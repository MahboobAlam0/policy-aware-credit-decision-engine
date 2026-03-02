#!/bin/bash
set -e

export API_URL="${API_URL:-http://localhost:8000/credit-decision}"

echo "Starting FastAPI backend on port 8000..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready
echo "Waiting for backend to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is ready."
        break
    fi
    sleep 1
done

echo "Starting Streamlit frontend on port 7860..."
streamlit run frontend.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none
