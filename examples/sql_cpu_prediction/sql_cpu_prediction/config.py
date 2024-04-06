## config.py

class Config:
    # Database connection string with a default value
    DB_CONNECTION_STRING: str = "sqlite:///sqlalchemy_example.db"
    
    # Path to save or load the model with a default value
    MODEL_PATH: str = "models/cpu_cost_predictor.pkl"
