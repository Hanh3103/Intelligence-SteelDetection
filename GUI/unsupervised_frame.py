from tkinter import messagebox

import customtkinter as ctk
import pandas as pd

from GUI.bar_plot import save_dataframe_as_csv
from api_functions.unsupervised import predict_data_from_file_unsupervised_kmeans
from api_functions.unsupervised import predict_data_from_file_unsupervised_affinity
from api_functions.unsupervised import predict_data_from_file_unsupervised_gmm
from api_functions.unsupervised import predict_data_from_file_unsupervised_hierarchical
from api_functions.unsupervised import predict_data_from_file_unsupervised_tsne
from api_functions.unsupervised import predict_data_from_file_unsupervised_pca
from api_functions.unsupervised import predict_data_from_file_unsupervised_umap
from enums import unsupervised_models_enum
from enums.unsupervised_models_enum import Models
from utils.upload_csv_or_xlsx import return_dataframe_filesize_filename_from_csv_or_xlsx


class UnsupervisedFrame(ctk.CTkFrame):

    def __init__(self, master, main_app_reference, **kwargs):
        super().__init__(master, **kwargs)
        self.main_app = main_app_reference

        self.configure(fg_color='#333333')
        self.columnconfigure(0, weight=1)

        '''Creation: unsupervised elements'''
        self.Unsupervised_label = ctk.CTkLabel(self, text="Unsupervised prediction",
                                               font=ctk.CTkFont(size=20, weight="bold"))

        self.algorithm_label = ctk.CTkLabel(self, text="Algorithm")

        unsupervised_algorithms = unsupervised_models_enum.Models.create_list()
        self.algorithm_var = ctk.StringVar(value=unsupervised_algorithms[0])
        self.algorithm_combobox = ctk.CTkOptionMenu(master=self,
                                                    dynamic_resizing=False,
                                                    width=170,
                                                    values=unsupervised_algorithms,
                                                    command=self.option_menu_callback_algorithm)
        self.algorithm_combobox.set(unsupervised_algorithms[0])

        self.komponents_slider_var = ctk.IntVar(value=5)
        self.komponents_slider = ctk.CTkSlider(master=self, from_=0, to=15,
                                               variable=self.komponents_slider_var, command=self.sliding)
        slider_text = 'number of clusters: ' + str(self.komponents_slider_var.get())
        self.komponents_slider_label = ctk.CTkLabel(master=self,
                                                    text=slider_text)

        self.predict_button = ctk.CTkButton(self, corner_radius=8,
                                            text='predict unsupervised', command=self.predict_unsupervised)

        self.unsupervised_data = ''

        self.upload_button = ctk.CTkButton(self, corner_radius=8,
                                           text='upload File', command=self.upload_file)

        self.neighbours_entry = ctk.CTkEntry(self,corner_radius=8, placeholder_text='number of neighbours')
        self.steps_entry = ctk.CTkEntry(self, corner_radius=8, placeholder_text='number of steps')
        self.min_dist_entry = ctk.CTkEntry(self, corner_radius=8, placeholder_text='Minimum distance for UMAP')

        '''Grid Config: unsupervised elements'''
        self.Unsupervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')
        self.upload_button.grid(row=1, column=0, padx=20, sticky='ew', columnspan=2)
        self.algorithm_label.grid(row=2, column=0, padx=(20, 0), pady=6, sticky='w')
        self.algorithm_combobox.grid(row=2, column=1, padx=(0, 20), pady=6, sticky='e')
        self.komponents_slider.grid(row=3, column=0, padx=20, pady=(6, 0), sticky='ew', columnspan=2)
        self.komponents_slider_label.grid(row=4, column=0, padx=20, sticky='ew', columnspan=2)
        self.predict_button.grid(row=8, column=0, padx=20, pady=6, sticky='ew', columnspan=2)

    def option_menu_callback_algorithm(self, choice):
        self.algorithm_var.set(choice)
        self.set_status()

    '''updates the unsupervised_slider_label with the current value of the slider'''
    def sliding(self, event=None):
        text = 'number of clusters: ' + str(self.komponents_slider_var.get())
        self.komponents_slider_label.configure(text=text)

    def predict_unsupervised(self):
        """
        When the Button 'Predict' in the category unsupervised is pressed it collects the algorithm which the user
        choose, the datafile and the different arguments which differ with each algorithm, to call the respected
        algorithm.
        """
        algorithm = Models.get_enum_by_value(self.algorithm_combobox.get())
        file = self.unsupervised_data

        komponents = self.komponents_slider_var.get()
        neighbours = self.neighbours_entry.get()
        steps = self.steps_entry.get()
        min_dist = self.min_dist_entry.get()

        prediction_data = unsupervised(algorithm, file, komponents, neighbours, steps, min_dist, '2D')
        if prediction_data is None:
            return
        save_dataframe_as_csv(prediction_data)
        self.main_app.create_unsupervised_canvas(prediction_data)

    '''calls the method: return_dataframe_filesize_filename_from_csv_or_xlsx() and saves return file in var: 
      unsupervised_data'''
    def upload_file(self):
        self.unsupervised_data = (return_dataframe_filesize_filename_from_csv_or_xlsx())

    def set_status(self):
        algorithm = Models.get_enum_by_value(self.algorithm_combobox.get())
        if (algorithm == unsupervised_models_enum.Models.KMEANS or
            algorithm == unsupervised_models_enum.Models.GMM or
            algorithm == unsupervised_models_enum.Models.hierarchical):
            self.steps_entry.grid_forget()
            self.min_dist_entry.grid_forget()
            self.neighbours_entry.grid_forget()
            self.komponents_slider.grid(row=3, column=0, padx=20, pady=(6, 0), sticky='ew', columnspan=2)
            self.komponents_slider_label.grid(row=4, column=0, padx=20, sticky='ew', columnspan=2)
        if (algorithm == unsupervised_models_enum.Models.pca or
                algorithm == unsupervised_models_enum.Models.affinity):
            self.steps_entry.grid_forget()
            self.min_dist_entry.grid_forget()
            self.neighbours_entry.grid_forget()
            self.komponents_slider.grid_forget()
            self.komponents_slider_label.grid_forget()
        if algorithm == unsupervised_models_enum.Models.tsne:
            self.steps_entry.grid(row=6, column=0, padx=20, pady=6, sticky='ew', columnspan=2)
            self.min_dist_entry.grid_forget()
            self.neighbours_entry.grid(row=5, column=0, padx=20, pady=6, sticky='ew', columnspan=2)
            self.komponents_slider.grid_forget()
            self.komponents_slider_label.grid_forget()
        if algorithm == unsupervised_models_enum.Models.umap:
            self.steps_entry.grid_forget()
            self.min_dist_entry.grid(row=7, column=0, padx=20, pady=6, sticky='ew', columnspan=2)
            self.neighbours_entry.grid(row=5, column=0, padx=20, pady=6, sticky='ew', columnspan=2)
            self.komponents_slider.grid(row=3, column=0, padx=20, pady=(6, 0), sticky='ew', columnspan=2)
            self.komponents_slider_label.grid(row=4, column=0, padx=20, sticky='ew', columnspan=2)

def unsupervised(model, file, komponents, neighbours, steps, min_dist, mode):
    if model == unsupervised_models_enum.Models.KMEANS:
        return predict_data_from_file_unsupervised_kmeans(file, komponents)
    elif model == unsupervised_models_enum.Models.GMM:
        return predict_data_from_file_unsupervised_gmm(file, komponents)
    elif model == unsupervised_models_enum.Models.hierarchical:
        return predict_data_from_file_unsupervised_hierarchical(file, komponents)
    elif model == unsupervised_models_enum.Models.tsne:
        try:
            neighbours = int(neighbours)
            steps = int(steps)
            if steps < 250:
                raise ValueError
            data = predict_data_from_file_unsupervised_tsne(file, neighbours, steps, mode)
            return pd.DataFrame(data[1:], columns=data[0])
        except ValueError:
            messagebox.showerror("Ungültige Eingabe", "Bitte geben Sie eine gültige Zahl ein. Steps Größer als 250")
    elif model == unsupervised_models_enum.Models.pca:
        data = predict_data_from_file_unsupervised_pca(file)
        return pd.DataFrame(data, columns=['Werte'])
    elif model == unsupervised_models_enum.Models.umap:
        try:
            neighbours = int(neighbours)
            min_dist = float(min_dist)
            data = predict_data_from_file_unsupervised_umap(file, neighbours, komponents, min_dist)
            return pd.DataFrame(data[1:], columns=data[0])
        except ValueError:
            messagebox.showerror("Ungültige Eingabe", "Bitte geben Sie eine gültige Zahl ein.")

    elif model == unsupervised_models_enum.Models.affinity:
        return predict_data_from_file_unsupervised_affinity(file)

