#!/bin/bash
# Launch script for PrexSyn Web Application

PORT=8501

# Check we're in the right directory
if [ ! -f "prexsyn_webapp.py" ]; then
    echo "Error: run this script from /home/christopher/git/prexsyn"
    exit 1
fi

# Check port availability
if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "Port $PORT is already in use (PID $(lsof -ti :$PORT))."
    echo "Kill it first:  kill \$(lsof -ti :$PORT)"
    exit 1
fi

# Activate conda environment
export PATH="/home/christopher/miniconda/bin:$PATH"
source /home/christopher/miniconda/etc/profile.d/conda.sh
conda activate prexsyn

# Start Streamlit in the background
streamlit run prexsyn_webapp.py \
    --server.port=$PORT \
    --server.address=localhost \
    --server.headless=true \
    --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

# Wait for the server to accept connections (up to 15 seconds)
echo "Starting PrexSyn..."
for i in $(seq 1 15); do
    if lsof -ti :$PORT >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! lsof -ti :$PORT >/dev/null 2>&1; then
    echo "Error: server did not start within 15 seconds."
    kill $STREAMLIT_PID 2>/dev/null
    exit 1
fi

echo "PrexSyn running at http://localhost:$PORT  (PID $STREAMLIT_PID)"
echo "Press Ctrl+C to stop."

# Open Chrome
google-chrome "http://localhost:$PORT" >/dev/null 2>&1 &

# Keep script alive so Ctrl+C kills the server
wait $STREAMLIT_PID
