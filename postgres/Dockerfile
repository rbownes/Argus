FROM postgres:15

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        postgresql-server-dev-15 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone pgvector repository
RUN git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git /tmp/pgvector

# Build and install pgvector
RUN cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Clean up build dependencies
RUN apt-get update \
    && apt-get remove -y --auto-remove \
        build-essential \
        postgresql-server-dev-15 \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
