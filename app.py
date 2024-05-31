import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np
import pandas as pd

class LaptopPricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laptop Price Predictor")

        # Load model and data
        self.model = pickle.load(open('best_gradient_boosting_model.pkl', 'rb'))
        self.df = pickle.load(open('df.pkl', 'rb'))

        # Create widgets
        self.company_label = ttk.Label(root, text="Company:")
        self.company_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["Company"].unique()])
        self.laptop_type_label = ttk.Label(root, text="Laptop Type:")
        self.laptop_type_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["TypeName"].unique()])
        self.ram_label = ttk.Label(root, text="RAM (GB):")
        self.ram_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["Ram"].unique()])
        self.weight_label = ttk.Label(root, text="Weight (kg):")
        self.weight_entry = ttk.Entry(root)
        self.touchscreen_label = ttk.Label(root, text="Touchscreen:")
        self.touchscreen_combobox = ttk.Combobox(root, values=["No", "Yes"])
        self.ips_label = ttk.Label(root, text="IPS Display:")
        self.ips_combobox = ttk.Combobox(root, values=["No", "Yes"])
        self.screen_size_label = ttk.Label(root, text="Screen Size (inches):")
        self.screen_size_entry = ttk.Entry(root)
        self.resolution_label = ttk.Label(root, text="Resolution:")
        self.resolution_combobox = ttk.Combobox(root, values=['1920x1080', '1366x768', '1600x900', '3840x2160', 
                                        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
        self.cpu_label = ttk.Label(root, text="CPU Brand:")
        self.cpu_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["CPU Brand"].unique()])
        self.hdd_label = ttk.Label(root, text="HDD (GB):")
        self.hdd_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["HDD"].unique()])
        self.ssd_label = ttk.Label(root, text="SSD (GB):")
        self.ssd_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["SSD"].unique()])
        self.gpu_label = ttk.Label(root, text="GPU Brand:")
        self.gpu_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["GPU Brand"].unique()])
        self.os_label = ttk.Label(root, text="Operating System:")
        self.os_combobox = ttk.Combobox(root, values=[str(x) for x in self.df["OS"].unique()])

        self.predict_button = ttk.Button(root, text="Predict", command=self.predict)

        # Grid layout
        self.company_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.company_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.laptop_type_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.laptop_type_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.ram_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.ram_combobox.grid(row=2, column=1, padx=5, pady=5)
        self.weight_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.weight_entry.grid(row=3, column=1, padx=5, pady=5)
        self.touchscreen_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.touchscreen_combobox.grid(row=4, column=1, padx=5, pady=5)
        self.ips_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.ips_combobox.grid(row=5, column=1, padx=5, pady=5)
        self.screen_size_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.screen_size_entry.grid(row=6, column=1, padx=5, pady=5)
        self.resolution_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.resolution_combobox.grid(row=7, column=1, padx=5, pady=5)
        self.cpu_label.grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.cpu_combobox.grid(row=8, column=1, padx=5, pady=5)
        self.hdd_label.grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.hdd_combobox.grid(row=9, column=1, padx=5, pady=5)
        self.ssd_label.grid(row=10, column=0, padx=5, pady=5, sticky="w")
        self.ssd_combobox.grid(row=10, column=1, padx=5, pady=5)
        self.gpu_label.grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.gpu_combobox.grid(row=11, column=1, padx=5, pady=5)
        self.os_label.grid(row=12, column=0, padx=5, pady=5, sticky="w")
        self.os_combobox.grid(row=12, column=1, padx=5, pady=5)

        self.predict_button.grid(row=13, columnspan=2, padx=5, pady=5)

    def predict(self):
        company = self.company_combobox.get()
        laptop_type = self.laptop_type_combobox.get()
        ram = self.ram_combobox.get()
        weight = self.weight_entry.get()
        touchscreen = 1 if self.touchscreen_combobox.get() == 'Yes' else 0
        ips = 1 if self.ips_combobox.get() == 'Yes' else 0
        screen_size = float(self.screen_size_entry.get())
        resolution = self.resolution_combobox.get()
        cpu = self.cpu_combobox.get()
        hdd = self.hdd_combobox.get()
        ssd = self.ssd_combobox.get()
        gpu = self.gpu_combobox.get()
        os = self.os_combobox.get()

        # Calculate PPI
        x_res, y_res = map(int, resolution.split('x'))
        ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size

        # Create a DataFrame from the input data
        data = pd.DataFrame({
            'Company': [company],
            'TypeName': [laptop_type],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen],
            'Ips': [ips],
            'PPI': [ppi],
            'CPU Brand': [cpu],
            'HDD': [hdd],
            'SSD': [ssd],
            'GPU Brand': [gpu],
            'OS': [os]
        })

        # Make prediction
        predicted_price = int(np.exp(self.model.predict(data)[0]))

        # Display result
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction Result")
        result_window.geometry("300x100")
        result_label = ttk.Label(result_window, text=f"The predicted Price of the laptop is {predicted_price}")
        result_label.grid(row=0, column=0, padx=10, pady=10)
        return predicted_price



def main():
    root = tk.Tk()
    app = LaptopPricePredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
