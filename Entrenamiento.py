# En este archivo se realiza el entrenamiento de la red neuronal
# para el reconocimiento de género. Se utiliza un dataset con imágenes
# de hombres y mujeres para el entrenamiento de la red neuronal.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Ruta al dataset
dataset_path = "DataSet/Modelo Final"

# Configuración de generador de datos
image_size = (128, 128)  # Tamaño al que se redimensionarán las imágenes
batch_size = 32 # Tamaño del batch

# Carga del dataset
def load_dataset():
    # Creación del generador
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # Normalizado de las imágenes

    # Carga del conjunto de entrenamiento
    train_generator = datagen.flow_from_directory(
        dataset_path, # Ruta al dataset
        target_size = image_size, # Tamaño de las imágenes
        batch_size = batch_size, # Tamaño del batch
        class_mode = "binary", # Clasificación binaria
        subset = "training" # Conjunto de entrenamiento
    )

    # Carga del conjunto de validación (test)
    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size = image_size, 
        batch_size = batch_size,
        class_mode = "binary",
        subset = "validation" # Conjunto de validación
    )

    return train_generator, validation_generator

# Construcción del modelo
def build_model(input_shape = (128, 128, 3)):
    model = Sequential([
        # Capa convolucional
        Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape),
        MaxPooling2D(pool_size = (2, 2)),

        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),

        # Aplanar las características
        Flatten(),

        # Capa totalmente conectada
        Dense(128, activation = 'relu'),
        Dropout(0.5),
        Dense(1, activation = 'sigmoid')  # Salida binaria (hombre o mujer)
    ])
    # Compilación del modelo
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

if __name__ == "__main__":
    # Carga de datos
    train_data, val_data = load_dataset()
    print("Clases: ", train_data.class_indices)
    print("Dataset cargado correctamente.")

    # Construcción del modelo
    model = build_model()
    model.summary()

    # Entrenamiento del modelo
    model.fit(train_data, validation_data = val_data, epochs = 10)

    # Guardar el modelo entrenado
    model.save("Modelos/reconocedor_generoMF.h5")
    print("Modelo guardado correctamente.")
