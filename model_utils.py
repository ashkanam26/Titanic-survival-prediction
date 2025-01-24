import joblib
def save_model(model, filename='titanic_model.pkl'):
    """save the trained model to a file"""
    joblib.dump(model, filename)
    print(f"model saved successfully as'{filename}'")

def load_model(filename='titanic_model.pkl'):
    """load the saved model from a file"""
    model = joblib.load(filename)
    print(f"model'{filename}'loaded successfully")
    return model
