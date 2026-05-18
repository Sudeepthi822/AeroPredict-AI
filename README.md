Industrial Predictive Maintenance System
🏭 Project Overview
This system is designed for smart factory environments to predict the Remaining Useful Life (RUL) of industrial assets, including robotic arms, automated assembly lines, and high-precision motors.

By analyzing multi-sensor data (such as torque, vibration, and thermal fluctuations), the model identifies subtle degradation patterns before they lead to costly factory-line stoppages.

🚀 Key Features
Predictive Analytics: Uses a Long Short-Term Memory (LSTM) Neural Network architecture to capture temporal dependencies in complex time-series sensor data.

Normalization Pipeline: Features a robust MinMaxScaler pre-processing layer to scale high-variance sensor data for gradient stability.

Integrity Validation: Custom security checks to verify data consistency and handle input validation before training.

Performance Evaluation: Integrated evaluation tracking using RMSE and MAE calculations for high-precision maintenance scheduling.

Asset Agnostic: Designed to seamlessly scale across various factory hardware and operational cycles.

📊 Technical Results
Mean Absolute Error (MAE): 16.53 cycles

Root Mean Squared Error (RMSE): 27.22 cycles

Training Duration: 100 Epochs with Adam optimization and Float32 precision

🛠 Setup & Usage
Ensure the dataset is present at: data/processed_train.csv

Install dependencies:

Bash
pip install -r docs/requirements.txt
Run training pipeline:

Bash
python train_lstm.py
Visualize performance results:

Bash
python visualize_results.py
📂 Project Structure
train_lstm.py: Model architecture, hyperparameter definitions, and training loops.

visualize_results.py: Evaluation metrics processing and loss/prediction curve graph generation.

preprocess_lstm.py: Data sequencing, rolling window execution, and integrity utilities.

docs/requirements.txt: Comprehensive list of project dependencies and environment configurations.

👥 Team: Zaalima Development
This project is a collaborative effort by the following team members:

@Sudeepthi822 (Data Scientist & ML Engineer)

Contributions: Core LSTM engine development, industrial robotics domain refactoring, time-series feature engineering, and production-ready script optimization.

@slowtypist (Lead Data Scientist & Frontend Architect)

Contributions: ML Model training, predict_engine.py architecture, React Dashboard development, and system design.

@Shivareddy8008 (Backend & Security Engineer)

Contributions: Backend API Bridge, Authentication logic, Database schema, and Input validation.
