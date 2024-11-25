from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Construction du modèle
def build_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),  # Couche dense avec 256 unités et activation ReLU
        BatchNormalization(),  # Normalisation par lots
        Dropout(0.5),  # Dropout pour éviter le surapprentissage
        Dense(128, activation='relu'),  # Couche dense avec 128 unités et activation ReLU
        BatchNormalization(),  # Normalisation par lots
        Dropout(0.5),  # Dropout pour éviter le surapprentissage
        Dense(64, activation='relu'),  # Couche dense avec 64 unités et activation ReLU
        BatchNormalization(),  # Normalisation par lots
        Dropout(0.5),  # Dropout pour éviter le surapprentissage
        Dense(32, activation='relu'),  # Couche dense avec 32 unités et activation ReLU
        BatchNormalization(),  # Normalisation par lots
        Dropout(0.5),  # Dropout pour éviter le surapprentissage
        Dense(1, activation='sigmoid')  # Couche de sortie avec activation sigmoïde
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])  # Compilation du modèle
    return model
