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

Please note: Even though the subgroups were in constant communication with each other to align on the project's vision, they still worked independently on their group projects. Therefore, if you want to run or work on the existing codebase, you might want to navigate to the relevant group's project directory and read any documentation or README files, if existing. Furthermore, each group has put their `requirements.txt` in the corresponding directory for you to be able to recreate their development environments.

If you want to expand the web application by adding more models, you can easily do so by adhering to the following process:

1. Create your own group project directory in the repository's root folder.
2. Develop your group's codebase to be able to train and predict with your chosen model.
    - For this step, you might be able to reuse a large proportion of the current code, e.g. the data loader source code.
3. Add a prediction API (`api.py`) to your group's directory which returns your model predictions when called.
    - Refer to other groups' prediction APIs when implementing yours.
4. Extend `WebApp/backend/main.py` by adding a prediction route which sends an HTTP request to your group's prediction API.
    - Follow the same pattern of the other predictions routes.
5. Add a `DOCKERFILE`, which details how your container should be built, to your group directory.
6. Extend `docker-compose.yaml` in the repository's root to create your container along the others when using `docker compose up`.
