import requests

token_save = None
token_type = None


def login_user(username, password):
    """
    Log in a user by sending a POST request to the token endpoint.

    Args:
        username (str): The username of the user attempting to log in.
        password (str): The password associated with the provided username.

    Raises:
        Exception: If the response status code is not 200.

    Returns:
        None. The function updates global variables `token_save` and `token_type`.

    Input:
        - `username` (str): The username of the user attempting to log in.
        - `password` (str): The password associated with the provided username.

    Output:
        - None. The function updates global variables `token_save` and `token_type`.
    """
    global token_save  # Declare global variable to store the access token
    global token_type  # Declare global variable to store the token type

    # Assign input parameters to local variables
    username = username
    password = password

    # Make a POST request to the token endpoint with provided credentials
    response = requests.post("http://127.0.0.1:8000/token", data={"username": username, "password": password})

    # Check if the response status code is not 200 (OK)
    if response.status_code != 200:
        # Raise an exception with the error detail from the response JSON
        raise Exception(response.json().get("detail"))
    else:
        # If the response status code is 200, extract and store the access token and token type
        response_json = response.json()
        token_save = response_json["access_token"]
        token_type = response_json["token_type"]


def register_user(username, password, full_name, email):
    """
    Register a user by sending a POST request to the specified endpoint.

    Args:
        username (str): The username for the new user.
        password (str): The password for the new user.
        full_name (str): The full name of the new user.
        email (str): The email address of the new user.

    Raises:
        SystemExit: If the HTTP request encounters an error.

    Returns:
        None.

    Input:
        - `username` (str): The username for the new user.
        - `password` (str): The password for the new user.
        - `full_name` (str): The full name of the new user.
        - `email` (str): The email address of the new user.

    Output:
        - None. The function prints the JSON response from the server.
    """
    # Assign input parameters to local variables
    username = username
    password = password
    full_name = full_name
    email = email

    # Create JSON data for the POST request
    json_data = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "password": password,
        "disabled": False
    }

    try:
        # Send a POST request to the registration endpoint
        response = requests.post("http://127.0.0.1:8000/registrate", json=json_data)

        # Check for HTTP errors
        response.raise_for_status()

        # Print the JSON response from the server
        print(response.json())
    except requests.exceptions.HTTPError as err:
        # Exit the system if an HTTP error occurs
        raise SystemExit(err)


def get_token():
    """
    Retrieve the stored access token.

    Returns:
        str: The access token stored in the global variable `token_save`.
    """
    return token_save


def get_token_type():
    """
    Retrieve the stored token type.

    Returns:
        str: The token type stored in the global variable `token_type`.
    """
    return token_type
