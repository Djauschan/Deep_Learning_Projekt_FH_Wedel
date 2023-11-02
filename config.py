import json

# Path to the JSON configuration file
config_file_path: str = "./config.json"

# The JSON configuration file is deserialized into a python dictionary.
config: dict = None

# Reading the JSON configuration file
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)
