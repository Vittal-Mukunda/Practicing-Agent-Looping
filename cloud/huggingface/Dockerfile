FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git unzip && \
    rm -rf /var/lib/apt/lists/*

# Kaggle CLI for data download
RUN pip install --no-cache-dir kaggle

# Clone project and install
RUN git clone https://github.com/Vittal-Mukunda/Practicing-Agent-Looping.git /app/cadvae
WORKDIR /app/cadvae
RUN pip install --no-cache-dir -e .

# Copy the sweep runner
COPY run_sweep.sh /app/run_sweep.sh
RUN chmod +x /app/run_sweep.sh

# Persistent storage mount point (HF Spaces)
RUN mkdir -p /data/results

CMD ["/app/run_sweep.sh"]
