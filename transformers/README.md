# Project Deep Learning - Transformer Group

This directory contains the source code of the transformer group. The members of the transformer group in the class of the wintersemester 2023/2024 are: Philipp, Niklas, Eira & Luca.

The codebase can be used to train transformer models for predicting time series, specifically stock prices. Furthermore, it also offers an interface to predict using the trained models and an API that returns the predictions in a format suitable for the web application to further process them. This API is started inside the transformers docker container which is built from the docker compose file in the project's root directory. The Dockerfile corresponding to the transformers docker container is also located here in this directory.

## Installation

If you want to run any of our code locally, you need to create an development environment and install the dependencies by following these steps:

1. Navigate into the transformer group's directory.
<br>`cd transformer`
2. Create a conda environment (python version 3.11 is preferred).
<br>`conda create -n transformer_env python==3.11`
3. Activate your environment.
<br>`conda activate transformer_env`
4. Install PyTorch.
<br>`pip install -r requirements_torch.txt`
5. Install the other requirements.
<br>`pip install -r requirements.txt`

If you just want to run a training or prediction pipeline or launch the tensorboard, you can avoid installing anything by just using our docker setup.

## Usage

To run the training and prediction pipeline locally, follow these steps:

1. Navigate into the transformer group's directory.
<br>`cd transformer`
2. Activate the environment you have created using your installation guide.
<br>`conda activate transformer_env`
3. Run a pipeline like this:
<br>`python -m src_transformers.main -c "path/to/config" -p train`
<br>Or like this:
<br>`python -m src_transformers.main --config "path/to/config" --pipeline train`
4. To run a prediction pipeline use the argument `predict` instead of `train`.

If you want to use our docker setup, navigate into this directory (`transformer/`). Afterwards, you can start different processes with the following commands:

- Start the training pipeline.
<br>`docker compose up train_transformer`
- Start the prediction pipeline.
<br>`docker compose up predict_transformer`
- Start the visualiasation utility script.
<br>`docker compose up visualize_data`
- Launch the tensorboard to look at your past training summaries.
<br>`docker compose up tensorboard`

NOTE: As the main goal of this project was to create a fully-fledged web application, we directly created a prediction API instead of a prediction pipeline. So there is no actual need to run the prediction pipeline as it will just print out a notification which refers you to the prediction API.

## Project Structure

- `data/`: The directory containing all data (input data for model training, output data such as scaler and model files, all configs used for our pipelines and miscellaneous data, e.g. the index mappings).
- `notebooks`: ?
- `scoring_functions/`: ?
- `src_transformers/`: Holds the actual codebase used for model training and prediction.
- `api.py`: The API that is started inside the transformers docker container. Requests to it are sent from the web application's backend and it returns the model predictions.
- `compose.yaml`: The docker compose file used to call the train and predict pipelines of this codebase. It can also be used to launch Tensorboard to visualize the training processes.
- `Dockerfile`: The Dockerfile used to build the transformers docker container.
- `README.md`: This file.
- `requirements_torch.txt`: Specifies the PyTorch requirements and the Index URL from which to download them from.
- `requirements.txt`: Specifies all other required python libraries.
- `setup.cfg`: The configuration we used for autopep8.
