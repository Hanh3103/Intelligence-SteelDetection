import configparser


def read_config():
    config = configparser.ConfigParser()
    config.read('resources/config.ini')
    return config


def read_config_section(section):
    config = read_config()
    return config[section]


def read_config_value(section, key):
    config = read_config()
    return config[section][key]
