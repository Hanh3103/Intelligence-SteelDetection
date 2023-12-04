from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import psycopg2
from psycopg2 import sql
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


class UserCreate(BaseModel):
    email: str
    password: str

templates = Jinja2Templates(directory="templates")

@app.get("/register", response_class=HTMLResponse)
def get_register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
def create_user(user: UserCreate, request: Request):
    try:
        connection = psycopg2.connect(
            host="localhost",
            port=5432,
            user="admin",
            password="pw",
            database="db"
        )

        cursor = connection.cursor()

        # Define the SQL query
        query = sql.SQL("INSERT INTO users (email, password) VALUES ({}, {});").format(
            sql.Literal(user.email),
            sql.Literal(user.password)
        )

        # Execute the query
        cursor.execute(query)

        # Commit the changes
        connection.commit()

        print("User created successfully!")

    except Exception as error:
        print(f"Error: {error}")
        raise HTTPException(status_code=500, detail=str(error))

    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()

    return templates.TemplateResponse("registration_success.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True, debug=True)