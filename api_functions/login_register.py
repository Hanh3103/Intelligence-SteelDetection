import requests

token_save = None
token_type = None


def login_user(username, password):
    global token_save  # Declare global variable
    global token_type  # Declare global variable

    username = username
    password = password

    response = requests.post("http://127.0.0.1:8000/token", data={"username": username, "password": password})

    if response.status_code != 200:
        raise Exception(response.json().get("detail"))
    else:
        response_json = response.json()
        token_save = response_json["access_token"]
        token_type = response_json["token_type"]


def register_user(username, password, full_name, email):
    username = username
    password = password
    full_name = full_name
    email = email
    json_data = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "password": password,
        "disabled": False
    }

    try:
        response = requests.post("http://127.0.0.1:8000/registrate", json=json_data)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def get_token():
    return token_save


def get_token_type():
    return token_type
