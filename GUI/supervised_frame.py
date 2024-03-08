import customtkinter as ctk
from api_functions.supervised import predict_with_supervised_algo
from utils.database_utils import get_dataframe_from_database


class SupervisedFrame(ctk.CTkFrame):
    def __init__(self, master, main_app_reference, **kwargs):
        super().__init__(master, **kwargs)
        self.main_app = main_app_reference

        '''Creation: supervised elements'''
        self.configure(fg_color='#333333')
        self.columnconfigure(0, weight=1)

        self.supervised_label = ctk.CTkLabel(self, text="Supervised",
                                             font=ctk.CTkFont(size=20, weight="bold"))

        self.supervised_model_var = ctk.StringVar(value="False")
        self.supervised_model_checkbox = ctk.CTkCheckBox(master=self, text="Model is existing",
                                                         variable=self.supervised_model_var, onvalue="True",
                                                         offvalue="False", command=self.change_model_status)

        self.supervised_button_Model = ctk.CTkButton(self, corner_radius=8,
                                                     text='trained Model', state="disabled")

        self.supervised_button_Dataframe = ctk.CTkButton(self, corner_radius=8,
                                                         text="Dataframe", state="disabled")

        self.supervised_button_encoder = ctk.CTkButton(self, corner_radius=8,
                                                       text="Labelencoder", state="disabled")

        self.unsupervised_algorithm_var = ctk.StringVar(value="xgboost_algorithm")
        self.supervised_algorithm_combobox = ctk.CTkOptionMenu(master=self,
                                                               values=["xgboost_algorithm", "deep_learning"],
                                                               command=self.option_menu_callback_algorithm)

        from GUI.gui import get_all_items_of_user_from_db_gui
        dataframes = list(get_all_items_of_user_from_db_gui())
        self.dataframe_var = ctk.StringVar(value=dataframes[0])
        self.dataframe_combobox = ctk.CTkOptionMenu(master=self,
                                                    values=dataframes,
                                                    command=self.option_menu_callback_dataframe)

        self.supervised_button_predict = ctk.CTkButton(self, corner_radius=8,
                                                       text='predict', command=self.predict_supervised)

        '''Grid Config: supervised elements'''
        self.supervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), sticky='ew')
        self.supervised_model_checkbox.grid(row=1, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_Model.grid(row=7, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_Dataframe.grid(row=4, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_encoder.grid(row=5, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_algorithm_combobox.grid(row=2, column=0, padx=20, pady=6, sticky='ew')
        self.dataframe_combobox.grid(row=3, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_predict.grid(row=8, column=0, padx=20, pady=6, sticky='ew')

    def change_model_status(self):
        if self.supervised_model_var.get() == "True":
            self.supervised_button_Model.configure(state="normal")
            self.supervised_button_Dataframe.configure(state="disabled")
            self.supervised_button_encoder.configure(state="disabled")
        else:
            self.supervised_button_Dataframe.configure(state="normal")
            self.supervised_button_encoder.configure(state="normal")
            self.supervised_button_Model.configure(state="disabled")

    def option_menu_callback_algorithm(self, choice):
        self.unsupervised_algorithm_var.set(choice)

    def predict_supervised(self):
        existing_model = self.supervised_model_var.get()
        algorithm = self.unsupervised_algorithm_var.get()
        dataframe_name = self.dataframe_var.get()
        if existing_model == "True":
            prediction = predict_with_supervised_algo(
                existing_model,
                "",
                algorithm,
                dataframe_name,
                "prediction")
        else:
            prediction = predict_with_supervised_algo(
                existing_model,
                "test_labeled_OES_data.csv",
                algorithm,
                dataframe_name,
                "prediction")
        prediction_data = get_dataframe_from_database(dataframe_name)
        self.main_app.create_canvas(prediction, prediction_data)

    def option_menu_callback_dataframe(self, choice):
        self.dataframe_var.set(choice)


