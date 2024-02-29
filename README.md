# Project Deep Learning - Trading Agent & Stock Predice Predictor

This repository contains the source code of a web application developed in the context of the module "Project Deep Learning" at the FH Wedel.
It was developed by the class of the winter semester 2023/2024. 

The web application is designed to be able to give a user recommendations for trading a specific selection of stocks. Furthermore, it can predict and visualize the prices of the stocks in question. The trading recommendations are given by a Reinforcement Learning agent and the stock prices are predicted using advanced machine learning and deep learning techniques, namely Gradient-Boosted Trees, Convolutional Neural Networks, and Transformers.

The class organized itself into the following subgroups:
- Frontend
- Backend
- Containerization
- Transformers
- CNN
- ANN (Machine Learning and Statistical methods)

## Installation

To get the application up and running, navigate to the root directory of the project and start it with the following command:

```
docker compose up
```

That's it! After a first installation, which might take a while as it collects and installs all dependencies in the containers, you can boot up the application in a matter of seconds.

## Usage

Open your web browser and navigate to `http://localhost:8080` to start using the application.

## Project Structure

- `ann/`: The project directory of the ANN group.
- `CNN/`: The project directory of the CNN group.
- `data/`: Contains the raw, unprocessed data that all groups use in their projects.
- `transformer/`: The project directory of the Transformer group.
- `WebApp/`: Contains the project directories of the Frontend and Backend groups.
- `.gitignore`: Global gitignore file for the whole project.
- `docker-compose.yaml`: The docker-compose file which starts all containers (frontend, backend, prediction APIs of each model group).
- `README.md`: This file.

## Contributing
