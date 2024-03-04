## Description of the Project
This project is a containerized application that combines a FastAPI Python backend with a PostgreSQL database. It focuses on the detection of various steel types through both supervised and unsupervised learning algorithms. The application includes functionalities such as unsupervised and supervised learning predictions, database interactions, and file uploads.

## Project Structure

- **api_functions:** Contains all functions related to call fastapi endpoints.
- **db:** Database-related modules and models.
- **endpoints:** FastAPI router modules for different functionalities.
- **gui:** Modules for the Tkinter-based graphical user interface.
- **MLModels:** Modules containing unsupervised and supervised learning algorithms.
- **utils:** Utility modules for various tasks.

## Docker Setup

### Dockerfile
The Dockerfile specifies the steps to build the Python environment for the application. It installs required dependencies listed in requirements.txt and copies the application code into the container.

### Docker Compose
The docker-compose.yml file defines services for PostgreSQL and the Python FastAPI app. It orchestrates these services, sets up environment variables, and ensures proper dependencies between containers.

## Usage

1. **Clone the repository:**
    
2. **Build and run the Docker containers:**
    ```bash
    docker-compose up --build
    ```
    This command builds the Docker image using the specified Dockerfile and starts the containers.

**Access the application:**

- FastAPI is accessible at http://localhost:8000.
- PostgreSQL is available at localhost:5432 with the specified credentials.

## Features

### Unsupervised Learning Predictions:

- Utilizes various algorithms for unsupervised learning predictions.
- Endpoints include kmeans, gmm, hierarchical, tsne, pca, and umap.

### Supervised Learning Predictions:

- Predicts data using supervised learning algorithms.
- Supports existing models and training new models.

### File Upload and Database Interaction:

- Uploads CSV or Excel files for data processing.
- Stores and retrieves data from a PostgreSQL database.

### Graphical User Interface (GUI):

- Provides a Tkinter-based GUI for a user-friendly experience.
- Allows interaction with the application features.

## Configuration

Configurations are stored in the `resources/config/config.ini` file. Update configuration parameters such as database credentials, token settings, and other application-specific values.

## Additional Notes

- Ensure Docker and Docker Compose are installed on your machine before running the application.

## Contributors

### Backend
- Luca
- Allan
- Hanh

### Frontend
- Lukas

### Supervised Learning Algorithms
- Julius
- Jona

### Unsupervised Learning Algorithms
- Joshua

