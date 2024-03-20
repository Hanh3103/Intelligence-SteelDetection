# SupervisedFrame Documentation

The `SupervisedFrame` class extends the custom tkinter frame (`ctk.CTkFrame`) to create a GUI component for supervised machine learning predictions within an application. This class facilitates the upload of data, selection of a machine learning algorithm, and the initiation of supervised predictions.

## Features

- **File Upload**: Users can upload data files directly into the database for subsequent predictions.
- **Model Selection**: Allows the selection between an existing model or training a new one based on the provided data.
- **Algorithm Choice**: Users can choose between predefined algorithms (e.g., XGBoost, Deep Learning) for performing the prediction.
- **Data Selection**: Separate dropdowns are provided for selecting data for prediction and, if a new model is to be trained, data for training.

## Class Initialization Parameters

- `master`: The parent tkinter widget.
- `main_app_reference`: Reference to the main application class to invoke other application-specific methods.
- **kwargs: Additional keyword arguments for the `ctk.CTkFrame`.

## Key Methods and Attributes

- `change_model_status()`: Toggles the availability of training data selection based on whether an existing model is used.
- `predict_supervised()`: Handles the prediction process, including the determination of whether to use an existing model or train a new one, and invokes the prediction algorithm.
- `option_menu_callback_algorithm(choice)`: Callback for the algorithm selection dropdown. Updates the chosen algorithm.
- `option_menu_callback_dataframe_to_prediction(choice)`: Callback for the data selection dropdown for prediction. Updates the selected data for prediction.
- `option_menu_callback_dataframe_to_train(choice)`: Callback for the data selection dropdown for training. Updates the selected data for training if a new model is to be trained.

## Future Work

-