import yaml
import os

# The path of the current file is determined.
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'data' directory located two levels up from the current file's directory
data_directory = os.path.join(
    current_file_directory, os.pardir, os.pardir, 'data')

# Construct the path to the YAML configuration file in the same directory
config_file_path = os.path.join(
    data_directory, 'test_configs', 'data_config.yaml')

# The YAML configuration file is deserialized into a python dictionary.
config: dict = None

# Reading the YAML configuration file
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
