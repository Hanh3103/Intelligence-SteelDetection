from utils.readConfig import read_config_value
from fastapi import HTTPException, status

try:
    # Read values from the configuration file using a utility function
    SECRET_KEY = read_config_value('TOKEN', 'SECRET_KEY')
    ALGORITHM = read_config_value('TOKEN', 'ALGORITHM')
    ACCESS_TOKEN_EXPIRE_MINUTES = int(read_config_value('TOKEN', 'ACCESS_TOKEN_EXPIRE_MINUTES'))
except Exception as e:
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error reading values from "
                                                                                  f"configuration file: {str(e)}")
