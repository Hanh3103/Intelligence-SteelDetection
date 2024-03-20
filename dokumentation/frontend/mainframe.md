# App Class Documentation

The `App` class in this Python module extends `customtkinter.CTk`, integrating various features to create a comprehensive user interface for both supervised and unsupervised prediction functionalities.

## Key Features

- **Dark Mode and Blue Theme**: Sets the default appearance to dark mode with a blue color theme using `customtkinter` settings.
- **Fullscreen Application Window**: Initializes the application window to fullscreen based on the user's screen resolution.
- **Sidebar Navigation**: Implements a sidebar for navigation between different functionalities like supervised and unsupervised prediction options.
- **Dynamic Content Area**: Based on user selection in the sidebar, dynamically switches between supervised and unsupervised frames in the content area.

## Class Initialization

- Inherits from `customtkinter.CTk`.
- Initializes the application with a title, fullscreen geometry, and binds a custom close window protocol.

## Main Components

### Sidebar

- **Static Sidebar**: Contains static elements like buttons for switching between supervised and unsupervised predictions.
- **Dynamic Sidebar**: Hosts either the supervised or unsupervised frame based on user selection, facilitating the respective prediction process.

### Supervised and Unsupervised Frames

- Dynamically loaded in the sidebar area based on user interaction, these frames (`SupervisedFrame`, `UnsupervisedFrame`) are designed to handle specific prediction tasks, including data upload and selection, algorithm choice, and visualization of prediction results.

### Visualization Canvas

- **Bar Plot Visualization**: Both supervised and unsupervised prediction results can be visualized in a bar plot format within the "Barplot" tab, utilizing the `matplotlib` integration with `customtkinter`.

## Methods

- `create_canvas()`: Creates a canvas for supervised prediction visualization based on provided data and predicted steel types.
- `create_unsupervised_canvas()`: Similar to `create_canvas`, but for unsupervised prediction data visualization.
- `show_supervised()`: Displays the supervised prediction frame and hides the unsupervised frame.
- `show_unsupervised()`: Displays the unsupervised prediction frame and hides the supervised frame.
- `on_close()`: Custom protocol for the window close event, ensuring graceful shutdown of related processes, including the FastAPI server.

