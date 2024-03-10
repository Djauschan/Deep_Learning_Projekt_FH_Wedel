# This file is used to read the config file and get the parameters from it.
import yaml

class Config_reader:
    """A class to read the config file and get the parameters from it. The config file must be a yaml file.
    """
    def __init__(self, config_file_path : str) -> None:
        """Constructor for the Config_reader class.
        Args:
            config_file_path (str): The path to the config file. The config file must be a yaml file.
        """
        self.config_file_path = config_file_path
        self.config = self.__read_config()

    def __read_config(self) -> dict:
        """Private method to read the config file.
        Returns:
            dict: The config file as a dictionary.
        """
        with open(self.config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def __get_nested_parameter(self, parameter_name : str, section_name : str) -> None:
        """Private method to get a nested parameter from the config file.
        Args:
            parameter_name (str): Name of the parameter which is nested inside the section of the config file.
            section_name (str): Name of the section in the config file.
        Raises:
            KeyError: If the parameter is not found in the config file.
        """
        try:
            return self.config[section_name][parameter_name]
        except KeyError:
            if section_name not in self.config:
                raise KeyError(f'Section "{section_name}" not found in config file ({self.config_file_path}).')
            raise KeyError(f'Parameter "{parameter_name}" not found under section "{section_name}" in config file ({self.config_file_path}).')

    def __get_single_parameter(self, parameter_name : str) -> object:
        """Private method to get a parameter from the config file.
        Args:
            parameter_name (str): Name of the parameter which is nested inside the section of the config file.
        Returns:
            object: the Value stored in the config file.
        """
        try: 
            return self.config[parameter_name]
        except KeyError:
            raise KeyError(f'Parameter "{parameter_name}" not found in config file ({self.config_file_path}).')

    def get_parameter(self, parameter_name : str, section_name : str = None) -> object:
        """Method to get a parameter from the config file.
        Args:
            parameter_name (str): Name of the parameter which is nested inside the section of the config file.
            section_name (str, optional): Name of the section in the config file. Defaults to None.
        Returns:
            object: the Value stored in the config file.
        """
        if section_name:
            return self.__get_nested_parameter(parameter_name, section_name)
        else:
            return self.__get_single_parameter(parameter_name)

    def __set_nested_parameter(self, new_value : object, parameter_name : str, section_name : str) -> None:
        """Private method to set a nested parameter in the config file.
        Args:
            new_value (object): The new value for the parameter.
            parameter_name (str): Name of the parameter which is nested inside the section of the config file.
            section_name (str): Name of the section in the config file.
        """
        if section_name not in self.config:
            self.config[section_name] = {}
        self.config[section_name][parameter_name] = new_value
        with open(self.config_file_path, 'w') as file:
            yaml.dump(self.config, file)


    def __set_single_parameter(self, new_value : object, parameter_name : str) -> None:
        """Private method to set a parameter in the config file.
        Args:
            parameter_name (str): Name of the parameter which is nested inside the section of the config file.
            new_value (object): The new value for the parameter.
        """
        self.config[parameter_name] = new_value
        with open(self.config_file_path, 'w') as file:
            yaml.dump(self.config, file)

    def set_parameter(self, value : object, parameter_name : str, section_name : str = None) -> None:
        """Method to set a parameter in the config file.
        Args:
            value (object): The new value for the parameter.
            parameter_name (str): Name of the parameter which is nested inside the section of the config file.
            section_name (str, optional): Name of the section in the config file. Defaults to None.
        """
        if section_name:
            self.__set_nested_parameter(value, parameter_name, section_name)
        else:
            self.__set_single_parameter(value, parameter_name)