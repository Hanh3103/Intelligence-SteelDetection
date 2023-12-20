import configparser


def readConfig():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def readConfigSection(section):
    config = readConfig()
    return config[section]


def readConfigValue(section, key):
    config = readConfig()
    return config[section][key]

