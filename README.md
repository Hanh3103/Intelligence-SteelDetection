# Intelligente-Stahlerkennung

> Quick setup
- when the docker desktop is not installed, please install the docker desktop
- Run docker desktop

````bash
$ git clone https://github.com/SchulzCode/Intelligente-Stahlerkennung.git
$ cd Intelligente-Stahlerkennung
$ docker-compose up -d
````

When you use Pycharm als your IDE
- Open the project using Pycharm
- Open the `docker-compose.yml` 
- Run this docker file

After the images are build and the container is set up. The api is listening at `http://127.0.0.1:8000 `
````bash
$ cd gui
$ python test.py
````