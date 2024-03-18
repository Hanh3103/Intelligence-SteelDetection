import customtkinter as ctk

from GUI.bar_plot import save_dataframe_as_csv
from api_functions.database import upload_csv_or_xlsx_to_db_table
from api_functions.supervised import predict_with_supervised_algo
from utils.database_utils import get_dataframe_from_database


class SupervisedFrame(ctk.CTkFrame):
    def __init__(self, master, main_app_reference, **kwargs):
        super().__init__(master, **kwargs)
        self.main_app = main_app_reference

        '''Creation: supervised elements'''
        self.configure(fg_color='#333333')
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=0)

        self.supervised_label = ctk.CTkLabel(self, text="Supervised prediction",
                                             font=ctk.CTkFont(size=20, weight="bold"))

        self.upload_button = ctk.CTkButton(self, corner_radius=8,
                                           text='upload file',
                                           command=upload_file_to_database)

        self.supervised_model_var = ctk.StringVar(value="False")
        self.model_checkbox = ctk.CTkCheckBox(master=self, text="Model is existing",
                                              variable=self.supervised_model_var, onvalue="True",
                                              offvalue="False", command=self.change_model_status)

        self.algorithm_label = ctk.CTkLabel(self, text="Algorithm")

        self.algorithm_var = ctk.StringVar(value="xgboost_algorithm")
        self.algorithm_combobox = ctk.CTkOptionMenu(master=self,
                                                    values=["xgboost_algorithm", "deep_learning"],
                                                    command=self.option_menu_callback_algorithm,
                                                    dynamic_resizing=False,
                                                    width=170)

        self.dataframe_to_prediction_label = ctk.CTkLabel(self, text="Data for prediction")

        from GUI.gui import get_all_items_of_user_from_db_gui
        dataframes = list(get_all_items_of_user_from_db_gui())
        if dataframes:
            self.dataframe_to_prediction_var = ctk.StringVar(value=dataframes[0])
        else:
            self.dataframe_to_prediction_var = ctk.StringVar(value="No data available")
        self.dataframe_to_prediction_combobox = ctk.CTkOptionMenu(master=self,
                                                                  dynamic_resizing=False,
                                                                  width=170,
                                                                  values=dataframes,
                                                                  command=self.option_menu_callback_dataframe_to_prediction)

        self.dataframe_to_train_label = ctk.CTkLabel(self, text="Data for training")

        if dataframes:
            self.dataframe_to_train_var = ctk.StringVar(value=dataframes[0])
        else:
            self.dataframe_to_train_var = ctk.StringVar(value="No data available")

        self.dataframe_to_train_combobox = ctk.CTkOptionMenu(master=self,
                                                             dynamic_resizing=False,
                                                             width=170,
                                                             values=dataframes,
                                                             command=self.option_menu_callback_dataframe_to_train)

        self.supervised_button_predict = ctk.CTkButton(self, corner_radius=8,
                                                       text='predict supervised', command=self.predict_supervised)

        '''Grid Config: supervised elements'''
        self.supervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), sticky='ew', columnspan=2)
        self.upload_button.grid(row=1, column=0, padx=20, pady=6, sticky='ew', columnspan=2)
        self.model_checkbox.grid(row=2, column=0, padx=20, pady=6, sticky='ew', columnspan=2)
        self.algorithm_label.grid(row=3, column=0, padx=(20, 0), pady=6, sticky='w')
        self.algorithm_combobox.grid(row=3, column=1, padx=(0, 20), pady=6, sticky='e')
        self.dataframe_to_prediction_label.grid(row=4, column=0, padx=(20, 0), pady=6, sticky='w')
        self.dataframe_to_prediction_combobox.grid(row=4, column=1, padx=(0, 20), pady=6, sticky='e')
        self.dataframe_to_train_label.grid(row=5, column=0, padx=(20, 0), pady=6, sticky='w')
        self.dataframe_to_train_combobox.grid(row=5, column=1, padx=(0, 20), pady=6, sticky='e')
        self.supervised_button_predict.grid(row=8, column=0, padx=20, pady=(6, 15), sticky='ew', columnspan=2)

    def change_model_status(self):
        if self.supervised_model_var.get() == "True":
            self.dataframe_to_train_combobox.configure(state='disabled')
        else:
            self.dataframe_to_train_combobox.configure(state='normal')

    def option_menu_callback_algorithm(self, choice):
        self.algorithm_var.set(choice)

    '''Im Backend wurden keine bool, sondern Strings genutzt. Methodenaufruf muss so ausgeführt werden'''
    def predict_supervised(self):
        existing_model = self.supervised_model_var.get()
        algorithm = self.algorithm_var.get()
        dataframe_to_predict = self.dataframe_to_prediction_var.get()
        dataframe_to_train = self.dataframe_to_train_var.get()
        if existing_model == "True":
            prediction = predict_with_supervised_algo(
                existing_model,
                "",
                algorithm,
                dataframe_to_predict,
                "prediction")
        else:
            prediction = predict_with_supervised_algo(
                existing_model,
                dataframe_to_train,
                algorithm,
                dataframe_to_predict,
                "prediction")
        prediction_data = get_dataframe_from_database(dataframe_to_predict)
        save_dataframe_as_csv(prediction_data)
        self.main_app.create_canvas(prediction, prediction_data)

    def option_menu_callback_dataframe_to_prediction(self, choice):
        self.dataframe_to_prediction_var.set(choice)

    def option_menu_callback_dataframe_to_train(self, choice):
        self.dataframe_to_train_var.set(choice)

def upload_file_to_database():
    upload_csv_or_xlsx_to_db_table()