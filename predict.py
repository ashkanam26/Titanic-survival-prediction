import pandas as pd

def predict_survival(model, passenger_info):
    """predict survival for a new passenger"""
    new_passenger = pd.DataFrame([passenger_info])
    prediction = model.predict(new_passenger)[0]
    return "survived" if prediction == 1 else "did not survived"