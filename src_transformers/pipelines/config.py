import yaml

# Path to the YAML configuration file
config_file_path: str = "./config.yaml"

# The YAML configuration file is deserialized into a python dictionary.
config: dict = None

# Reading the YAML configuration file
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
