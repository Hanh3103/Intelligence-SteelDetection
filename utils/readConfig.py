import configparser
import os


def read_config():
    # Function to read the entire configuration from 'config.ini'
    config = configparser.ConfigParser()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        path_of_config = os.path.join(parent_dir, 'resources', 'config', 'config.ini')
        config.read(path_of_config)
    except configparser.Error as e:
        print(f"Error reading 'config.ini': {e}")
        # You can choose to raise an exception or return a default configuration here
        return None
    return config


def read_config_section(section):
    # Function to read a specific section from the configuration
    config = read_config()
    if config:
        try:
            return config[section]
        except KeyError as e:
            print(f"Error: Section '{section}' not found in 'config.ini'")
    return None


def read_config_value(section, key):
    # Function to read a specific value from the configuration
    config = read_config()
    if config:
        try:
            return config[section][key]
        except KeyError as e:
            print(f"Error: Key '{key}' not found in section '{section}' of 'config.ini'")
    return None
