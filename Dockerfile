# Use the official image as a parent image.
FROM python:3.7

ARG PROJ_REPO=https://github.com/dli-stats/distributed_cox.git
ARG GIT_TAG=repro
ARG DATA_REPO=https://github.com/dli-stats/distributed_cox_paper_simudata.git

ENV REPO_DIR=/distributed_cox
ENV DATA_DIR=/cox-data
ENV DATA_NPZ_DIR=/distributed_cox/npz_data

# Clone paper simulated data
RUN git clone ${DATA_REPO} ${DATA_DIR}

# Prepare repo code
WORKDIR ${REPO_DIR}
RUN git clone --depth=1 --branch ${GIT_TAG} ${PROJ_REPO} ${REPO_DIR}

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install -e .

# Preprocess the raw csv data into .npz files
RUN mkdir -p $DATA_NPZ_DIR
RUN python scripts/distributed/convert_simulated_data.py \
  convert_from_csv \
  "${DATA_DIR}/dat_std_simulated.csv" \
  ${DATA_NPZ_DIR}
