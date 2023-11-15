import yaml
import os

# Get the directory of the current script
current_script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the YAML configuration file in the same directory
config_file_path = os.path.join(current_script_directory, 'config.yaml')

# The YAML configuration file is deserialized into a python dictionary.
config: dict = None

# Reading the YAML configuration file
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
