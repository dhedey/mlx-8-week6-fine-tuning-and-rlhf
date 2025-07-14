# Use a uv Python image as a base
# NOTE: PyArrow (a dependency of streamlit) doesn't provide pre-built wheels for alpine
# and has complex dependencies for building from source, so we use a bookworm base image instead
# https://github.com/apache/arrow/issues/39846#issuecomment-1916269760
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt-get update && apt-get install -y curl

#####################################################
# ================= APP CONTAINER ================= #
#####################################################
FROM base AS app

# Set the working directory in the container
WORKDIR /app

# Copy just the files relevant to syncing to leverage the Docker cache
COPY uv.lock pyproject.toml /app/

# Install the dependencies
RUN uv sync --frozen --no-dev

# Expose the port that Streamlit runs on
EXPOSE 8505

# Add a healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8505/_stcore/health

# Copy the rest of current directory contents across
# Note that importantly the .venv directory is excluded in the .dockerignore file
COPY . /app

# Command to run the Streamlit app
ENTRYPOINT ["uv", "run", "streamlit", "run", "streamlit/app.py", "--server.port=8505"]
