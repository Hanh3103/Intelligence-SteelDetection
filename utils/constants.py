from utils.readConfig import read_config_value

SECRET_KEY = read_config_value('TOKEN', 'SECRET_KEY')
ALGORITHM = read_config_value('TOKEN', 'ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(read_config_value('TOKEN', 'ACCESS_TOKEN_EXPIRE_MINUTES'))
