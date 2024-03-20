# UnsupervisedFrame Documentation

`UnsupervisedFrame` is a custom frame class in a GUI application that facilitates unsupervised machine learning predictions. This class inherits from `ctk.CTkFrame` provided by the `customtkinter` library, offering a tailored appearance and functionality suited for unsupervised learning tasks.

## Features

- **Algorithm Selection**: Users can select an unsupervised learning algorithm from a dropdown menu. The selection dynamically adjusts available options and inputs based on the chosen algorithm.

- **Parameter Configuration**: Depending on the selected algorithm, users can adjust various parameters such as the number of clusters, the number of neighbors, the number of steps, and the minimum distance for UMAP.

- **Data Upload**: Provides a button to upload data files (CSV or XLSX), which will be used for making predictions.

- **Prediction Execution**: A 'Predict' button initiates the prediction process with the selected algorithm and parameters on the uploaded data file.

## Key Components and Methods

- `__init__`: Initializes the frame, creating the GUI components and configuring the layout.

- `option_menu_callback_algorithm`: Callback function for the algorithm selection dropdown. It updates the UI based on the selected algorithm.

- `sliding`: Updates labels based on slider values (e.g., the number of clusters).

- `predict_unsupervised`: Collects input data, parameters, and the selected algorithm to perform unsupervised prediction. It displays the prediction results in the application.

- `upload_file`: Invokes a dialog for the user to upload a data file. The selected file's path is stored for subsequent analysis.

- `set_status`: Adjusts the visibility and availability of UI components based on the currently selected algorithm to ensure that only relevant options are presented to the user.

## Supported Algorithms

`UnsupervisedFrame` supports various unsupervised learning algorithms, including:

- K-Means
- Gaussian Mixture Model (GMM)
- Hierarchical Clustering
- t-SNE
- PCA (Principal Component Analysis)
- UMAP (Uniform Manifold Approximation and Projection)
- Affinity Propagation

Each algorithm has its specific requirements and parameters, which are handled dynamically in the UI based on the user's selection.

## Future Work

- **Input Validation While Typing**: Instant validation of user inputs while the user is typing would improve the user experience by providing immediate feedback. This can reduce input errors and make the overall guidance more intuitive.

- **Simplification and Expert Mode**: The current setup could be optimized by simplifying the user interface and the options available for a more general user base. For advanced users, an expert mode could be introduced that unlocks advanced settings and parameterizations. This would make the application appealing to both beginners and experienced users.
