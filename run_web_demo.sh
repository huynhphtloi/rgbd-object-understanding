#!/bin/bash
# Start Streamlit web demo

cd "$(dirname "$0")/.."

echo "Starting RGB-D Scene Understanding Web Demo..."
echo "Open: http://localhost:8501"
echo ""

streamlit run src/app/streamlit_app.py --logger.level=info
