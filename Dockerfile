# ─────────────────────────────────────────────────────────────────────────────
# NovaProspect — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Uses conda-forge's mambaforge base to get THREAD-SAFE HDF5 + netCDF4.
# This fixes the SIGSEGV crashes caused by multi-threaded HDF5 access
# (H5F_addr_decode segfault in ThreadPoolExecutor workers).
#
# Build:   docker compose build
# Run:     docker compose up -d
# Attach:  docker exec -it novaprospect bash
# ─────────────────────────────────────────────────────────────────────────────

FROM condaforge/mambaforge:latest AS base

# System deps for GDAL, rasterio, and geospatial libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    tmux \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Create the conda environment with all geospatial deps from conda-forge.
# conda-forge builds HDF5 with --enable-threadsafe, which is the whole point.
RUN mamba create -n nova -c conda-forge -y \
    python=3.12 \
    numpy \
    scipy \
    pandas \
    geopandas \
    rasterio \
    pyproj \
    shapely \
    netcdf4 \
    hdf5 \
    h5py \
    duckdb \
    boto3 \
    pillow \
    requests \
    python-dotenv \
    matplotlib \
    laspy \
    lazrs-python \
    pyogrio \
    && mamba clean -afy

# Activate the env by default for all RUN / CMD / ENTRYPOINT
ENV PATH=/opt/conda/envs/nova/bin:$PATH
ENV CONDA_DEFAULT_ENV=nova

WORKDIR /app

# Install pip-only packages (openai SDK, osmium, anything not on conda-forge)
COPY requirements-pip.txt .
RUN pip install --no-cache-dir -r requirements-pip.txt

# Copy project code
COPY . .

# Default: drop into bash with the env active
CMD ["bash"]
