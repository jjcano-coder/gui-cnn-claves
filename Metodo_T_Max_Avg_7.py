import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import serial
import math

# Definición de la capa personalizada T-Max-Avg (sin cambios)
class TMaxAvgPooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), threshold=0.8, **kwargs):
        super(TMaxAvgPooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.threshold = threshold

    def call(self, inputs):
        max_pool = tf.nn.max_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding="VALID")
        avg_pool = tf.nn.avg_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding="VALID")
        result = tf.where(max_pool >= self.threshold, max_pool, avg_pool)
        return result

    def get_config(self):
        config = super(TMaxAvgPooling2D, self).get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "threshold": self.threshold,
        })
        return config

class OCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento de Caracteres OCR")

        # Inicialización de parámetros (sin cambios)
        self.prediction_count = 0
        self.max_predictions = 4
        self.learning_rate = 0.001
        self.epochs = 5
        self.threshold = 0.8

        # Ventana de predicciones (sin cambios)
        self.predictions_window = tk.Toplevel(self)
        self.predictions_window.title("Configuración de Predicciones")
        self.max_predictions_label = tk.Label(
            self.predictions_window, text=f"Máximo de predicciones: {self.max_predictions}", padx=10, pady=10
        )
        self.max_predictions_label.pack()

        self.increase_predictions_button = tk.Button(
            self.predictions_window, text="+ Predicciones", command=self.increase_max_predictions
        )
        self.increase_predictions_button.pack(side=tk.LEFT)

        self.decrease_predictions_button = tk.Button(
            self.predictions_window, text="- Predicciones", command=self.decrease_max_predictions
        )
        self.decrease_predictions_button.pack(side=tk.LEFT)

        # Ventana de coeficiente de aprendizaje (sin cambios)
        self.lr_window = tk.Toplevel(self)
        self.lr_window.title("Seguimiento del Coeficiente de Aprendizaje")
        self.lr_label = tk.Label(
            self.lr_window, text=f"Coeficiente de aprendizaje: {self.learning_rate:.4f}", padx=10, pady=10
        )
        self.lr_label.pack()
        self.increase_lr_button = tk.Button(
            self.lr_window, text="Aumentar LR", command=self.increase_learning_rate
        )
        self.increase_lr_button.pack(side=tk.LEFT)
        self.decrease_lr_button = tk.Button(
            self.lr_window, text="Disminuir LR", command=self.decrease_learning_rate
        )
        self.decrease_lr_button.pack(side=tk.LEFT)

        # Ventana de épocas (sin cambios)
        self.epochs_window = tk.Toplevel(self)
        self.epochs_window.title("Seguimiento de Épocas")
        self.epochs_label_window = tk.Label(
            self.epochs_window, text=f"Número de épocas: {self.epochs}", padx=10, pady=10
        )
        self.epochs_label_window.pack()
        self.increase_epochs_button = tk.Button(
            self.epochs_window, text="+ Épocas", command=self.increase_epochs
        )
        self.increase_epochs_button.pack(side=tk.LEFT)
        self.decrease_epochs_button = tk.Button(
            self.epochs_window, text="- Épocas", command=self.decrease_epochs
        )
        self.decrease_epochs_button.pack(side=tk.LEFT)

        # Ventana del umbral T-Max-Avg (sin cambios)
        self.threshold_window = tk.Toplevel(self)
        self.threshold_window.title("Seguimiento del Umbral T-Max-Avg")
        self.threshold_label_window = tk.Label(
            self.threshold_window, text=f"Umbral T-Max-Avg: {self.threshold:.2f}", padx=10, pady=10
        )
        self.threshold_label_window.pack()
        self.increase_threshold_button = tk.Button(
            self.threshold_window, text="Aumentar Umbral", command=self.increase_threshold
        )
        self.increase_threshold_button.pack(side=tk.LEFT)
        self.decrease_threshold_button = tk.Button(
            self.threshold_window, text="Disminuir Umbral", command=self.decrease_threshold
        )
        self.decrease_threshold_button.pack(side=tk.LEFT)

        # Canvas y botones principales (sin cambios)
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()

        self.train_button = tk.Button(self, text="Entrenar", command=self.train_model)
        self.train_button.pack(side=tk.LEFT)
        self.predict_button = tk.Button(self, text="Predecir", command=self.predict)
        self.predict_button.pack(side=tk.LEFT)
        self.clear_button = tk.Button(self, text="Borrar", command=self.clear)
        self.clear_button.pack(side=tk.LEFT)

        self.epochs_label = tk.Label(self, text=f"Épocas: {self.epochs}")
        self.epochs_label.pack(side=tk.LEFT)

        self.result_label = tk.Label(self, text="Resultado: ")
        self.result_label.pack(side=tk.RIGHT)

        self.count_label = tk.Label(self, text="Conteo de predicciones: 0")
        self.count_label.pack(side=tk.RIGHT)

        # Configuración de Arduino (sin cambios)
        self.arduino = serial.Serial('COM7', 9600)
        self.image = Image.new("L", (28, 28), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.model = None

    def draw_lines(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+8, y+8, fill='black', width=10)
        self.draw.ellipse([x//10, y//10, x//10 + 1, y//10 + 1], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 255)
        self.draw = ImageDraw.Draw(self.image)

    def update_predictions_window(self):
        self.max_predictions_label.config(text=f"Máximo de predicciones: {self.max_predictions}")

    def increase_max_predictions(self):
        self.max_predictions += 1
        self.update_predictions_window()

    def decrease_max_predictions(self):
        if self.max_predictions > 1:
            self.max_predictions -= 1
            self.update_predictions_window()

    def update_lr_window(self):
        self.lr_label.config(text=f"Coeficiente de aprendizaje: {self.learning_rate:.4f}")

    def increase_learning_rate(self):
        self.learning_rate += 0.0001
        self.update_lr_window()

    def decrease_learning_rate(self):
        self.learning_rate = max(0.0001, self.learning_rate - 0.0001)
        self.update_lr_window()

    def update_epochs_window(self):
        self.epochs_label_window.config(text=f"Número de épocas: {self.epochs}")
        self.epochs_label.config(text=f"Épocas: {self.epochs}")

    def increase_epochs(self):
        self.epochs += 1
        self.update_epochs_window()

    def decrease_epochs(self):
        if self.epochs > 1:
            self.epochs -= 1
            self.update_epochs_window()

    def increase_threshold(self):
        self.threshold += 0.05
        self.threshold_label_window.config(text=f"Umbral T-Max-Avg: {self.threshold:.2f}")

    def decrease_threshold(self):
        if self.threshold > 0.05:
            self.threshold -= 0.05
            self.threshold_label_window.config(text=f"Umbral T-Max-Avg: {self.threshold:.2f}")

    def train_model(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        self.model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            TMaxAvgPooling2D(threshold=self.threshold),
            Conv2D(64, (3, 3), activation="relu"),
            TMaxAvgPooling2D(threshold=self.threshold),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss="categorical_crossentropy", metrics=["accuracy"])

        history = self.model.fit(x_train, y_train, epochs=self.epochs, validation_data=(x_test, y_test))

        # Crear gráficas usando solo Tkinter (sin cambios)
        self.create_tkinter_plots(history)

        train_accuracy = history.history['accuracy'][-1]
        test_accuracy = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        test_loss = history.history['val_loss'][-1]

        # Mostrar resultados en ventanas (corregido: "Pérdida" en lugar de "MSE")
        train_results_window = tk.Toplevel(self)
        train_results_window.title("Resultados del Entrenamiento")
        train_results_label = tk.Label(train_results_window, 
                                     text=f"Precisión de entrenamiento: {train_accuracy:.2%}\nPérdida de entrenamiento: {train_loss:.4f}", 
                                     padx=10, pady=10)
        train_results_label.pack()

        test_results_window = tk.Toplevel(self)
        test_results_window.title("Resultados de Prueba")
        test_results_label = tk.Label(test_results_window, 
                                    text=f"Precisión de prueba: {test_accuracy:.2%}\nPérdida de prueba: {test_loss:.4f}", 
                                    padx=10, pady=10)
        test_results_label.pack()

    def create_tkinter_plots(self, history):
        # Crear ventana para las gráficas (sin cambios)
        plots_window = tk.Toplevel(self)
        plots_window.title("Gráficas de Entrenamiento")
        
        # Configurar tamaño de las gráficas (sin cambios)
        graph_width = 600
        graph_height = 350
        padding = 50
        
        # Crear canvas para la gráfica de exactitud (sin cambios)
        accuracy_canvas = tk.Canvas(plots_window, width=graph_width, height=graph_height, bg='white')
        accuracy_canvas.pack()
        
        # Crear canvas para la gráfica de pérdida (sin cambios)
        loss_canvas = tk.Canvas(plots_window, width=graph_width, height=graph_height, bg='white')
        loss_canvas.pack()
        
        # Dibujar ejes (sin cambios)
        self.draw_axes(accuracy_canvas, graph_width, graph_height, padding, "Época", "Exactitud", y_max=1.0)
        self.draw_axes(loss_canvas, graph_width, graph_height, padding, "Época", "Pérdida", y_max=max(history.history['loss'] + history.history['val_loss'])*1.1)
        
        # Dibujar curvas de exactitud (sin cambios)
        self.plot_data(accuracy_canvas, history.history['accuracy'], 'blue', graph_width, graph_height, padding, y_max=1.0)
        self.plot_data(accuracy_canvas, history.history['val_accuracy'], 'red', graph_width, graph_height, padding, y_max=1.0)
        
        # Dibujar curvas de pérdida (sin cambios)
        self.plot_data(loss_canvas, history.history['loss'], 'blue', graph_width, graph_height, padding)
        self.plot_data(loss_canvas, history.history['val_loss'], 'red', graph_width, graph_height, padding)
        
        # Añadir leyenda (sin cambios)
        legend_x = graph_width - 120
        legend_y = 20
        
        accuracy_canvas.create_text(legend_x, legend_y, text="Entrenamiento", fill='blue', anchor='w', font=('Arial', 9))
        accuracy_canvas.create_text(legend_x, legend_y + 20, text="Validación", fill='red', anchor='w', font=('Arial', 9))
        
        loss_canvas.create_text(legend_x, legend_y, text="Entrenamiento", fill='blue', anchor='w', font=('Arial', 9))
        loss_canvas.create_text(legend_x, legend_y + 20, text="Validación", fill='red', anchor='w', font=('Arial', 9))

    def draw_axes(self, canvas, width, height, padding, x_label, y_label, y_max=1.0):
        # Dibujar ejes (sin cambios)
        canvas.create_line(padding, height-padding, width-padding, height-padding, width=2)
        canvas.create_line(padding, height-padding, padding, padding, width=2)
        
        canvas.create_text(width//2, height-padding//2 + 15, text=x_label)
        canvas.create_text(padding + 5, padding//2, text=y_label, anchor='w', font=('Arial', 10, 'bold'))
        
        for i in range(0, 11):
            y = height - padding - (height - 2*padding) * (i/10)
            canvas.create_line(padding-5, y, padding, y)
            canvas.create_text(padding-10, y, text=f"{i/10*y_max:.1f}", anchor='e', font=('Arial', 8))
        
        max_ticks = 10
        step = max(1, math.ceil(self.epochs / max_ticks))
        
        for i in range(0, self.epochs, step):
            x = padding + (width - 2*padding) * (i / (self.epochs-1)) if self.epochs > 1 else padding
            canvas.create_line(x, height-padding, x, height-padding+5)
            canvas.create_text(x, height-padding+15, text=str(i), anchor='n', font=('Arial', 8))
        
        if self.epochs > 1 and (self.epochs-1) % step != 0:
            x = width - padding
            canvas.create_line(x, height-padding, x, height-padding+5)
            canvas.create_text(x, height-padding+15, text=str(self.epochs-1), anchor='n', font=('Arial', 8))

    def plot_data(self, canvas, data, color, width, height, padding, y_max=None):
        if not y_max:
            y_max = max(data) * 1.1
        
        points = []
        for i, value in enumerate(data):
            x = padding + (width - 2*padding) * (i / (len(data)-1))
            y = height - padding - (height - 2*padding) * (value / y_max)
            points.extend([x, y])
        
        if len(points) > 2:
            canvas.create_line(*points, fill=color, width=2)
        
        for i in range(0, len(points), 2):
            canvas.create_oval(points[i]-3, points[i+1]-3, points[i]+3, points[i+1]+3, fill=color)

    def calculate_metrics_per_digit(self, digit, x_test, y_test):
        predictions = self.model.predict(x_test)
        mask = np.argmax(y_test, axis=1) == digit
        digit_predictions = predictions[mask]
        digit_labels = y_test[mask]
        mse = np.mean(np.square(digit_labels - digit_predictions))
        accuracy = np.mean(np.argmax(digit_predictions, axis=1) == np.argmax(digit_labels, axis=1))
        return {"mse": mse, "accuracy": accuracy}

    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img).astype("float32") / 255
        img = img.reshape(1, 28, 28, 1)

        if self.model:
            prediction = np.argmax(self.model.predict(img))
            self.result_label.config(text=f"Resultado: {prediction}")

            self.prediction_count += 1
            self.count_label.config(text=f"Conteo de predicciones: {self.prediction_count}")

            self.arduino.write(str(prediction).encode())

            # Calcular métricas para el dígito predicho (con nota sobre MSE)
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255
            y_test = to_categorical(y_test, 10)
            metrics = self.calculate_metrics_per_digit(prediction, x_test, y_test)

            # Mostrar ventana con métricas (añadida nota sobre MSE)
            digit_window = tk.Toplevel(self)
            digit_window.title(f"Resultados para el dígito {prediction}")
            digit_label = tk.Label(
                digit_window,
                # text=f"Precisión: {metrics['accuracy']:.2%}\nMSE: {metrics['mse']:.4f}",
                text=f"Precisión para el dígito {prediction}: {metrics['accuracy']:.2%}",
                padx=10,
                pady=10,
            )
            digit_label.pack()

            if self.prediction_count >= self.max_predictions:
                self.prediction_count = 0
                self.clear()

if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()
