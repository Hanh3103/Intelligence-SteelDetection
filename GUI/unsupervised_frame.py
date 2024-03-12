import customtkinter as ctk

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
        self.Unsupervised_label = ctk.CTkLabel(self, text="Unsupervised",
                                               font=ctk.CTkFont(size=20, weight="bold"))

        self.algorithm_label = ctk.CTkLabel(self, text="Algorithm")

        unsupervised_algorithms = unsupervised_models_enum.Models.create_list()
        self.unsupervised_algorithm_var = ctk.StringVar(value=unsupervised_algorithms[0])
        self.unsupervised_algorithm_combobox = ctk.CTkOptionMenu(master=self,
                                                                 dynamic_resizing=False,
                                                                 width=170,
                                                                 values=unsupervised_algorithms,
                                                                 command=self.option_menu_callback_algorithm)
        self.unsupervised_algorithm_combobox.set(unsupervised_algorithms[0])

        self.unsupervised_slider_var = ctk.IntVar(value=5)
        self.unsupervised_slider = ctk.CTkSlider(master=self, from_=0, to=15,
                                                 variable=self.unsupervised_slider_var, command=self.sliding)
        slider_text = 'number of clusters: ' + str(self.unsupervised_slider_var.get())
        self.unsupervised_slider_label = ctk.CTkLabel(master=self,
                                                      text=slider_text)

        self.unsupervised_button_predict = ctk.CTkButton(self, corner_radius=8,
                                                         text='predict unsupervised', command=self.predict_unsupervised)

        self.unsupervised_data = ''

        self.neighbours_entry = ctk.CTkEntry(self,corner_radius=8, placeholder_text='number of neighbours')

        '''Grid Config: unsupervised elements'''
        self.Unsupervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')
        self.algorithm_label.grid(row=1, column=0, padx=(20, 0), pady=6, sticky='w')
        self.unsupervised_algorithm_combobox.grid(row=1, column=1, padx=(0, 20), pady=6, sticky='e')
        self.unsupervised_slider.grid(row=2, column=0, padx=20, pady=(6, 0), sticky='ew', columnspan=2)
        self.unsupervised_slider_label.grid(row=3, column=0, padx=20, sticky='ew', columnspan=2)
        self.unsupervised_button_predict.grid(row=4, column=0, padx=20, pady=6, sticky='ew', columnspan=2)
        self.neighbours_entry.grid(row=4, column=0, padx=20, pady=6, sticky='ew', columnspan=2)

    def option_menu_callback_algorithm(self, choice):
        self.unsupervised_algorithm_var.set(choice)

    '''updates the unsupervised_slider_label with the current value of the slider'''
    def sliding(self, event=None):
        text = 'number of clusters: ' + str(self.unsupervised_slider_var.get())
        self.unsupervised_slider_label.configure(text=text)

    def predict_unsupervised(self):
        """
        When the Button 'Predict' in the category unsupervised is pressed it collects the algorithm which the user
        choose, the datafile and the different arguments which differ with each algorithm, to call the respected
        algorithm.
        """
        algorithm = Models.get_enum_by_value(self.unsupervised_algorithm_combobox.get())
        file = self.unsupervised_data
        komponents = self.unsupervised_slider_var.get()
        unsupervised(algorithm, file, komponents)

    '''calls the method: return_dataframe_filesize_filename_from_csv_or_xlsx() and saves return file in var: 
      unsupervised_data'''
    def upload_file(self):
        self.unsupervised_data = (return_dataframe_filesize_filename_from_csv_or_xlsx())


def unsupervised(model, file, komponents):
    if model == unsupervised_models_enum.Models.KMEANS:
        predict_data_from_file_unsupervised_kmeans(file, komponents)
    elif model == unsupervised_models_enum.Models.GMM:
        predict_data_from_file_unsupervised_gmm(file, komponents)
    elif model == unsupervised_models_enum.Models.hierarchical:
        predict_data_from_file_unsupervised_hierarchical(file, komponents)
    elif model == unsupervised_models_enum.Models.tsne:
        predict_data_from_file_unsupervised_tsne(file, ...)
    elif model == unsupervised_models_enum.Models.pca:
        predict_data_from_file_unsupervised_pca(file)
    elif model == unsupervised_models_enum.Models.umap:
        predict_data_from_file_unsupervised_umap(file, ...)
    elif model == unsupervised_models_enum.Models.affinity:
        predict_data_from_file_unsupervised_affinity(file)
