import configparser


def read_config():
    # Function to read the entire configuration from 'config.ini'
    config = configparser.ConfigParser()
    config.read('resources/config.ini')
    return config


def read_config_section(section):
    # Function to read a specific section from the configuration
    config = read_config()
    return config[section]


def read_config_value(section, key):
    # Function to read a specific value from the configuration
    config = read_config()
    return config[section][key]
