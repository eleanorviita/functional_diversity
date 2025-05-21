FROM ubuntu:latest
LABEL authors="elean"

ENTRYPOINT ["top", "-b"]

FROM continuumio/miniconda3

# Install Psi4 and dependencies from Conda
RUN conda install -c conda-forge psi4 numpy rdkit matplotlib pandas scipy openpyxl functools -y

# Set the working directory
WORKDIR /workspace

# Set Conda environment variables
ENV PATH="/opt/conda/bin:$PATH"

# Default command
CMD ["/bin/bash"]

ENV OMP_NUM_THREADS=8



# Switch to root to install dependencies
#USER root

# Update package manager and install Conda if necessary
#RUN apt-get update && apt-get install -y python3-pip python3-venv


# Ensure Conda is initialized properly
#RUN conda init bash

# Create and activate a virtual environment
#RUN python3 -m venv /opt/venv
#ENV PATH="/opt/venv/bin:$PATH"

# Install Psi4 and required packages using Conda
#RUN conda install -c conda-forge psi4 numpy rdkit matplotlib pandas scipy -y


# Set default interpreter path
#ENV PYTHONPATH="/opt/conda/bin/python3"

#CMD ["/bin/bash"]

