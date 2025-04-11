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
      
    def update_cutoff_visibility(self, event=None):
        tipo = self.filter_type.get()
        if tipo == "pasa-banda":
            self.high_cutoff_label.grid()
            self.high_cutoff_entry.grid()
        else:
            self.high_cutoff_label.grid_remove()
            self.high_cutoff_entry.grid_remove()

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.sample_rate, data = wavfile.read(file_path)
            self.audio_data = data if data.ndim == 1 else data[:, 0]  # Mono
            self.filtered_data = None
            self.plot_signal(self.audio_data, self.axs[0], "Señal Original")
            self.axs[1].cla()
            self.axs[1].set_title("Señal Filtrada")
            self.canvas.draw()

    def apply_filter(self):
        if self.audio_data is None:
            messagebox.showwarning("Advertencia", "Primero carga un archivo de audio.")
            return

        try:
            order = int(self.order_entry.get())
            filter_type = self.filter_type.get()
            nyq = 0.5 * self.sample_rate

            if filter_type == "pasa-bajas":
                cutoff = float(self.cutoff_entry.get())
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
            elif filter_type == "pasa-altas":
                cutoff = float(self.cutoff_entry.get())
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='high', analog=False)
            elif filter_type == "pasa-banda":
                low = float(self.cutoff_entry.get()) / nyq
                high = float(self.high_cutoff_entry.get()) / nyq
                b, a = butter(order, [low, high], btype='band', analog=False)
            else:
                messagebox.showerror("Error", "Tipo de filtro desconocido.")
                return

            self.filtered_data = filtfilt(b, a, self.audio_data)
            self.plot_signal(self.filtered_data, self.axs[1], f"Señal Filtrada ({filter_type})")
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Error", "Parámetros inválidos.")

    def show_fft(self):
        if self.audio_data is None:
            messagebox.showwarning("Advertencia", "Primero carga un archivo.")
            return

        fft_vals_orig = np.fft.fft(self.audio_data)
        fft_freqs = np.fft.fftfreq(len(self.audio_data), 1 / self.sample_rate)

        plt.figure()
        plt.title("Transformada de Fourier")

        plt.plot(fft_freqs[:len(fft_vals_orig)//2], np.abs(fft_vals_orig[:len(fft_vals_orig)//2]), label="Original")

        if self.filtered_data is not None:
            fft_vals_filtered = np.fft.fft(self.filtered_data)
            plt.plot(fft_freqs[:len(fft_vals_filtered)//2], np.abs(fft_vals_filtered[:len(fft_vals_filtered)//2]), label="Filtrada")

        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def save_audio(self):
        if self.filtered_data is None:
            messagebox.showwarning("Advertencia", "No hay señal procesada para guardar.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if file_path:
            wavfile.write(file_path, self.sample_rate, self.filtered_data.astype(np.int16))
            messagebox.showinfo("Guardado", f"Archivo guardado en {file_path}")

    def plot_signal(self, signal, axis, title):
        axis.cla()
        axis.plot(np.linspace(0, len(signal)/self.sample_rate, num=len(signal)), signal)
        axis.set_title(title)
        axis.set_xlabel("Tiempo (s)")
        axis.set_ylabel("Amplitud")
        axis.grid(True)


# Ejecutar la app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()
