import os
import signal

from fastapi import HTTPException
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from GUI.bar_plot import BarPlot
from GUI.supervised_frame import SupervisedFrame
from GUI.unsupervised_frame import UnsupervisedFrame

import customtkinter as ctk
from api_functions.database import get_all_items_of_user_from_db, upload_csv_or_xlsx_to_db_table, \
    delete_db_table_by_name


class App(ctk.CTk):

    # Set appearance mode and default color theme
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    def __init__(self):
        super().__init__()

# Set login status
        self.login_successful = False

# root
        self.title("Steel Identifier.py")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

# root Layout
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

# Mainframe creation
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.grid(row=0, column=1, sticky="nsew")

        self.tabview = ctk.CTkTabview(self.main_frame, fg_color='transparent')
        self.tabview.grid(row=0, column=0, sticky="nsew")
        self.tabview.add("Barplot")
        self.this = ctk.CTk
        self.tabview.tab("Barplot").grid(sticky='nsew')

        self.canvas = None

# Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0, width=350)
        self.sidebar_frame.grid(row=0, column=0, sticky="ns")
        self.sidebar_frame.grid_propagate(False)
        self.sidebar_frame.columnconfigure(0, weight=1)

        '''Creation: Sidebar elements'''
        self.static_sidebar_frame = ctk.CTkFrame(self.sidebar_frame)
        self.dynamic_sidebar_frame = ctk.CTkFrame(self.sidebar_frame)

        '''Grid Config: Sidebar elements'''
        self.static_sidebar_frame.columnconfigure(0, weight=1)
        self.static_sidebar_frame.grid(row=0, column=0, padx=(0, 6), pady=(0, 12), sticky='nsew')
        self.dynamic_sidebar_frame.grid(row=1, column=0, padx=(0, 6), sticky='nsew')
        self.dynamic_sidebar_frame.columnconfigure(0, weight=1)

# Static  Sidebar
        '''Creation: static elements'''
        self.logo_label = ctk.CTkLabel(self.static_sidebar_frame, text="Side Bar",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.sidebar_button_supervised = ctk.CTkButton(self.static_sidebar_frame, corner_radius=8,
                                                       text='Supervised prediction', command=self.show_supervised)
        self.sidebar_button_unsupervised = ctk.CTkButton(self.static_sidebar_frame, corner_radius=8,
                                                         text='Unsupervised prediction', command=self.show_unsupervised)

        '''Grid Config: static elements'''
        self.logo_label.grid(row=0, column=0, padx=6, pady=(15, 6))
        self.sidebar_button_supervised.grid(row=2, column=0, padx=20, pady=6, sticky='nsew')
        self.sidebar_button_unsupervised.grid(row=3, column=0, padx=20, pady=(6, 15), sticky='nsew')

# Dynamic Sidebar
        # ------------------ Unsupervised ------------------------------

        self.unsupervised_frame = UnsupervisedFrame(self.dynamic_sidebar_frame, self)

        # ------------------ Supervised ---------------------------------

        self.supervised_frame = SupervisedFrame(self.dynamic_sidebar_frame, self)
        self.supervised_frame.grid(row=1, column=0, sticky='nsew')

    '''This method creates a canvas to display a bar plot of element percentages in various steel types using the 
    provided predicted steel types and prediction data'''
    def create_canvas(self, predicted_stell_types, prediction_data):
        bar_plot = BarPlot()
        elemente = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Al', 'Cu']
        fig = bar_plot.create_supervised_plot(predicted_stell_types, elemente, prediction_data)
        self.canvas = FigureCanvasTkAgg(fig, master=self.tabview.tab("Barplot"))
        self.canvas.draw()
        self.canvas.get_tk_widget().place(relx=0.0, rely=0.0, width=self.tabview.tab("Barplot").winfo_width(),
                                          height=self.tabview.tab("Barplot").winfo_height())

    def create_unsupervised_canvas(self, prediction_data):
        bar_plot = BarPlot()
        fig = bar_plot.create_unsupervised_plot(prediction_data)
        self.canvas = FigureCanvasTkAgg(fig, master=self.tabview.tab("Barplot"))
        self.canvas.draw()
        self.canvas.get_tk_widget().place(relx=0.0, rely=0.0, width=self.tabview.tab("Barplot").winfo_width(),
                                          height=self.tabview.tab("Barplot").winfo_height())

    '''displays the supervised frame in the dynamic part of the Sidebar and hides the unsupervised part'''
    def show_supervised(self):
        if self.unsupervised_frame.winfo_ismapped():
            self.supervised_frame.grid(row=0, column=0, sticky='nsew')
            self.unsupervised_frame.grid_forget()

    '''displays the unsupervised frame in the dynamic part of the Sidebar and hides the supervised part'''
    def show_unsupervised(self):
        if self.supervised_frame.winfo_ismapped():
            self.unsupervised_frame.grid(row=0, column=0, sticky='nsew')
            self.supervised_frame.grid_forget()

    def on_close(self):
        stop_fastapi_server()
        self.destroy()

        
def get_all_items_of_user_from_db_gui():
    return get_all_items_of_user_from_db()


def delete_db_table_by_name_gui():
    delete_db_table_by_name("luca300_labeled_OES_data.csv")


def stop_fastapi_server():
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error shutting down FastAPI server: {str(e)}")
