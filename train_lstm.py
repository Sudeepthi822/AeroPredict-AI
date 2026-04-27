import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from preprocess_lstm import create_sequences, verify_data_integrity
import os

def train_lstm_model():
    # 1. Security Check (Your unique addition!)
    data_path = 'data/processed_train.csv'
    if os.path.exists(data_path):
        verify_data_integrity(data_path)
    else:
        print("Error: processed_train.csv not found!")
        return

    # 2. Load and Shape Data
    df = pd.read_csv(data_path)
    
    # We create sequences (windows of 30 cycles)
    # LSTMs need this 3D shape to "remember" patterns
    X_sequences = create_sequences(df, sequence_length=30)
    
    # For simplicity, we'll match the target (y) to the sequences
    y = df['RUL'].values[30:] 

    # 3. Build the LSTM Architecture
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_sequences.shape[1], X_sequences.shape[2])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1) # Predicting the Remaining Useful Life
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 4. Train the Model
    print("Training LSTM Neural Network... this is more intensive than Random Forest.")
    model.fit(X_sequences, y, epochs=10, batch_size=32, validation_split=0.1)

    # 5. Save the Advanced Model
    model.save('lstm_model.h5')
    print("Success! 'lstm_model.h5' created.")

if __name__ == "__main__":
    train_lstm_model()