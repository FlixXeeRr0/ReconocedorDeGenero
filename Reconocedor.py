# Proyecto para reconocimiento de genero entre hombres y mujeres
# mediante el uso de redes neuronales convolucionales (CNN).
# Se debera de validar la imagen agregandola al directorio de imagenes o mediante la camapra del dispositivo

# La obtencion de los datasets se obtuvo de las siguientes ligas:
# Dataset 1: https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women/data
# Este dataset cuenta con 835 imagenes de mujeres y 833 imagenes de hombres.
# Este dataset se utilozo para el entrenamiento del modelo 1.
# Accuracy: 0.9861 - Loss: 0.0643 - val_accuracy: 0.6697 - val_loss: 1.3036

# Dataset 2: https://www.kaggle.com/datasets/playlist/men-women-classification
# Este dataset cuenta con 1,912 imagenes de mujeres y 1,418 imagenes de hombres.
# Este dataset se utilizo para el entrenamiento del modelo 2.
# Accuracy: 0.9616 - Loss: 0.1274 - val_accuracy: 0.6490 - val_loss: 1.0799

# La combinacion de ambos datasets se utilizo para el entrenamiento del modelo final.
# Con un total de 2,747 imagenes de mujeres y 2,251 imagenes de hombres.
# Accuracy: 0.9890 - Loss: 0.0364 - val_accuracy: 0.7688 - val_loss: 1.1043


# Instalacion de librerias necesarias agregadas en el archivo Requisitos.txt
# pip install -r Requisitos.txt

# Librerias para la interfaz grafica
import tkinter as tk
from tkinter import filedialog, messagebox
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon
# Librerias para el reconocimiento de genero
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model_path = "Modelos/reconocedor_generoMF.h5"
model = load_model(model_path)

# Tamaño esperado de las imágenes
image_size = (128, 128)

# Función para reconocimiento de una imagen
def predict_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", f"No se pudo cargar la imagen: {image_path}")
        return
    image = cv2.resize(image, image_size)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalizar la imagen

    # Realizar predicción
    prediction = model.predict(image)[0][0]
    if prediction > 0.5:
        return "MUJER"
    else:
        return "HOMBRE"

# Funcion para reconocimiento en tiempo real
def real_time_recognition():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar y normalizar el frame
        resized_frame = cv2.resize(frame, image_size)
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Predicción
        prediction = model.predict(input_frame)[0][0]
        gender = "Mujer" if prediction > 0.5 else "Hombre"

        text = gender
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.25
        font_thickness = 2

        frame_height, frame_width = frame.shape[:2]

        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = 10
        text_y = frame_height - 10

        cv2.rectangle(
            frame,
            (0, text_y - text_size[1] - 10),
            (frame_width, text_y + 10),
            (204, 136, 0),
            -1
        )
        
        # Obtener tamaño del texto
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size

        text_x = (frame_width - text_width) // 2 
        text_y = frame_height - 15

        # Dibujar el texto
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        top_text = 'Presione la tecla "Q" para salir'
        font_scale = 0.75
        text_top_size, _ = cv2.getTextSize(top_text, font, font_scale, font_thickness)
        text_width, text_height = text_top_size
        top_text_x = (frame_width - text_width) // 2 
        top_text_y = text_height + 15

        # Dibujar el texto centrado en la parte superior
        cv2.putText(frame, top_text, (top_text_x, top_text_y), font, font_scale, (255, 255, 255), font_thickness)

        cv2.imshow("Reconocimiento de Genero", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Interfaz gráfica
def open_main_window():
    # Funciones para los eventos de los botones
    def on_image_recognition():
        # Abrir explorador de archivos para seleccionar la imagen
        image_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")],
        )
        if image_path:
            result = predict_image(image_path)
            messagebox.showinfo("Resultado del analisis", f"Género predecido: {result}")
        
    def on_real_time_recognition():
        real_time_recognition()

    # Creacion de la ventana principal
    app = QApplication([])

    window = QWidget()
    window.setWindowTitle("Reconocimiento de Género")
    window.setStyleSheet("background-color: #242428;")
    layout = QVBoxLayout()

    # Estilo CSS para los botones y el encabezado
    label_style = """
        QLabel {
            font-family: Arial;
            font-size: 20px;
            font-weight: bold;
            color: white;
            margin-top: 10px;
            width: 100%;
            text-align: center;
        }
    """
    button_style = """
        QPushButton {
            background-color: #004466;
            color: white;
            border: 2px solid #004466;
            border-radius: 10px;
            padding: 10px 20px;
            text-align: left;
            width: 100%;
        }
        QPushButton:hover {
            background-color: #0088cc;
            border: 2px solid white;
        }
    """

    data_btn_imagen = {
        "Text": "Por imagen",
        "Icon": "./public/imagenColor.png",
        "event": on_image_recognition
    }
    data_btn_realtime = {
        "Text": "En tiempo real",
        "Icon": "./public/camaraColor.png",
        "event": on_real_time_recognition
    }

    # Función para crear un botón con texto e ícono
    def create_button(text, icon_path):
        button = QPushButton(text)
        button.setFont(QFont("Arial", 12, QFont.Bold))
        button.setStyleSheet(button_style)
        button.setIcon(QIcon(icon_path))
        button.setIconSize(QSize(30, 30))
        button.setLayoutDirection(Qt.RightToLeft)
        return button

    # Creacion de etiqueta como encabezado
    label_encabezado = QLabel("Método de reconocimiento")
    label_encabezado.setStyleSheet(label_style)

    # Creacion de botones
    btn_imagen = create_button(data_btn_imagen["Text"], data_btn_imagen["Icon"])
    btn_imagen.clicked.connect(lambda: data_btn_imagen["event"]())
    btn_realtime = create_button(data_btn_realtime["Text"], data_btn_realtime["Icon"])
    btn_realtime.clicked.connect(lambda: data_btn_realtime["event"]())

    label_developer = QLabel("Desarrollador: Agustin Cardoza Perez")
    label_developer.setStyleSheet("font-family: Arial; font-size: 16px; color: white; font-weight: 500;")

    # Añadir el botón al layout
    layout.addWidget(label_encabezado)
    layout.addSpacing(20)
    layout.addWidget(btn_imagen)
    layout.addSpacing(20)
    layout.addWidget(btn_realtime)
    layout.addSpacing(20)
    layout.addWidget(label_developer)
    layout.addSpacing(10)

    layout.setAlignment(Qt.AlignCenter)

    # Configurar ventana principal
    window.setLayout(layout)
    window.setFixedSize(350, 250)
    window.show()

    # Ejecutar la aplicación
    app.exec_()


# Función principal
if __name__ == "__main__":
    open_main_window()

