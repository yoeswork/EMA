## Introduction

# Installation

conda create -n ema python=3.10
conda activate ema
pip install -r requirements.txt

# Usage

<!-- Add the port number to uvicorn -->
uvicorn ema:app --reload


# Testing

python ema.py