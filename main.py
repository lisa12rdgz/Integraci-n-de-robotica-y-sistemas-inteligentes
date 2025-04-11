import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SignalProcessorApp:
  def __init__(self, root):
        self.root = root
        self.root.title("HMI - Procesamiento de Señales de Audio")
        self.audio_data = None
        self.sample_rate = None
        self.filtered_data = None

        self.create_widgets()

    def create_widgets(self):
        # Botones
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Cargar archivo", command=self.load_audio).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="Aplicar Filtro", command=self.apply_filter).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="Transformada Fourier", command=self.show_fft).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(btn_frame, text="Guardar Resultado", command=self.save_audio).grid(row=0, column=3, padx=5, pady=5)

        # Parámetros del filtro
        filter_frame = tk.LabelFrame(self.root, text="Parámetros del Filtro")
        filter_frame.pack(pady=10)

        # Menú desplegable para tipo de filtro
        tk.Label(filter_frame, text="Tipo de filtro:").grid(row=0, column=0, padx=5, pady=5)
        self.filter_type = ttk.Combobox(filter_frame, values=["pasa-bajas", "pasa-altas", "pasa-banda"], state="readonly")
        self.filter_type.current(0)
        self.filter_type.grid(row=0, column=1, padx=5)
        self.filter_type.bind("<<ComboboxSelected>>", self.update_cutoff_visibility)

        # Entradas para frecuencias de corte
        tk.Label(filter_frame, text="Frecuencia de corte (Hz):").grid(row=1, column=0, padx=5, pady=5)
        self.cutoff_entry = tk.Entry(filter_frame)
        self.cutoff_entry.insert(0, "1000")
        self.cutoff_entry.grid(row=1, column=1)

        self.high_cutoff_label = tk.Label(filter_frame, text="Frecuencia de corte alta (Hz):")
        self.high_cutoff_label.grid(row=1, column=2, padx=5, pady=5)
        self.high_cutoff_entry = tk.Entry(filter_frame)
        self.high_cutoff_entry.insert(0, "3000")
        self.high_cutoff_entry.grid(row=1, column=3)

        # Orden del filtro
        tk.Label(filter_frame, text="Orden del filtro:").grid(row=2, column=0, padx=5)
        self.order_entry = tk.Entry(filter_frame)
        self.order_entry.insert(0, "4")
        self.order_entry.grid(row=2, column=1)

        self.update_cutoff_visibility()  # Ajusta visibilidad inicial

        # Área de gráficos
        self.fig, self.axs = plt.subplots(2, 1, figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

# Ejecutar la app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()
