FROM python:3.11-slim

# System deps (libgomp1 needed by LightGBM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py frontend.py ./
COPY data/ data/
COPY training/ training/
COPY policy/ policy/
COPY explanations/ explanations/
COPY monitoring/ monitoring/
COPY serving/ serving/

# Copy only the essential artifact (the coupled bundle)
COPY artifacts/xai_loandefault/coupled_model_explainer.pkl \
    artifacts/xai_loandefault/coupled_model_explainer.pkl

# Copy entrypoint
COPY start.sh .
RUN chmod +x start.sh

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces exposes port 7860
EXPOSE 7860

CMD ["./start.sh"]
