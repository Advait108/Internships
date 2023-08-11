import joblib
file_vec = joblib.load('vectorizer.joblib')
file_xgb = joblib.load('xgb_model.joblib')

reference = {
    "unknown": 0, "payment": 1, "pictures": 2, "sales": 3, "human": 4,
    "cancel": 5, "tech": 6, "policy": 7, "bye": 8, "hello": 9, "claims": 10
}

def classify_message(message):
    # Vectorize
    message_vectorized = vectorizer.transform([message])
    # Predict
    xgb_probabilities = xgb_classifier.predict_proba(message_vectorized)[0]

    # Predicted probabilities
    xgb_results = [(key, probability) for key, probability in zip(reference.keys(), xgb_probabilities)]
    xgb_results.sort(key=lambda x: x[1], reverse=True)

    return xgb_results

while True:
    input_message = input("Enter a message: ")
    result = classify_message(input_message)
    print("")
    print("XGBoost Classification Results:")
    for class_label, probability in result:
        print(class_label, ": ", probability)
    print("")