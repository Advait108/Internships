import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load the service account key JSON file
cred = credentials.Certificate('C:/Users/advai/Desktop/clupp internship/auto-clupp-firebase-adminsdk-evw67-b6eb6f3b75.json')

# Initialize the Firebase Admin SDK with the service account credentials
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Reference the 'IT/WhatsApp/predictions' collection in Firestore
collection_ref = db.collection('IT').document('whatsApp').collection('predictions')

# Retrieve all documents from the 'IT/WhatsApp/predictions' collection
documents = collection_ref.get()

# expressions to filter out
expressions_buffers = {
    'hola': 11,
    'buen': 20,
    'que tal': 3,
    'si': 10,
    'ok': 10,
    'acuerdo': 20,
    'gracias': 20,
    'no': 10,
    'hi': 11
}
# check for expression method
def filter(string):
    for expression,buffer in expressions_buffers.items():
        if expression.lower() in string.lower() and len(string) < (len(expression) + buffer):
            return True
    return False

distribution = [0,0,0,0,0,0,0,0,0,0,0]
# making dict
db = {}
deleted_list = []
for document in documents:
    if 'choice' in document.to_dict():
        choice = document.to_dict()['choice']
    else:
        reference = {
            0: "unknown", 1: "payment", 2: "pictures", 3: "sales", 4: "human",
            5: "cancel", 6: "tech", 7: "policy", 8: "bye", 9: "hello", 10:
                "claims"
        }
        antiref = {
            "unknown": 0, "payment": 1, "pictures": 2, "sales": 3, "human": 4,
            "cancel": 5, "tech": 6, "policy": 7, "bye": 8, "hello": 9,
            "claims": 10
        }

        predictions = document.to_dict()['prediction']
        entries = predictions[0]["structValue"]["fields"]["confidences"]\
            ["listValue"]["values"]
        float_values = []
        for entry in entries:
            float_values.append(entry["numberValue"])
        index = float_values.index(max(float_values))


        choice = reference[index]


    try:
        distribution[antiref[choice]] = distribution[antiref[choice]] + 1

        # filtering out unneeded data
        if filter(document.to_dict()['message']):
            deleted_list.append(document.to_dict()['message'])
            continue
        #filling out dict
        doc_id = document.id
        message = document.to_dict()['message']
        timestamp = document.to_dict()['timestamp']
        timeraw = timestamp / 1000
        timee = time.ctime(timeraw)

        db[doc_id] = {
            'message': message,
            'choice': choice,
            'timestamp': timee
        }

    except:
        continue

print(distribution)

# Splitting the data into training and testing sets
# Stratify parameter ensures even distribution of classes in train and test sets
messages = []
choices = []
for ID, info in db.items():
    messages.append(info['message'])
    choices.append(info['choice'])

X_train, X_test, y_train, y_test = train_test_split(
    messages, choices, test_size=0.30, stratify=choices, random_state=42
)

#vectorizing labels
reference = {
    "unknown": 0, "payment": 1, "pictures": 2, "sales": 3, "human": 4,
    "cancel": 5, "tech": 6, "policy": 7, "bye": 8, "hello": 9, "claims": 10
}
y_train_numeric = []
for label in y_train:
    for key, value in reference.items():
        if label == key:
            y_train_numeric.append(value)
            break

y_test_numeric = []
for label in y_test:
    for key, value in reference.items():
        if label == key:
            y_test_numeric.append(value)
            break

y_train = y_train_numeric
y_test = y_test_numeric

#vectorizing messages
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

X_train = X_train_vectorized
X_test = X_test_vectorized

# Creating and training the random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicting on the test set using the random forest classifier
rf_predictions = rf_classifier.predict(X_test)

# Calculating the accuracy and F1 score of the random forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_f1_score = f1_score(y_test, rf_predictions, average='weighted')
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest F1 Score:", rf_f1_score)

# Creating and training the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Predicting on the test set using the XGBoost classifier
xgb_predictions = xgb_classifier.predict(X_test)

# Calculating the accuracy and F1 score of the XGBoost model
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_f1_score = f1_score(y_test, xgb_predictions, average='weighted')
print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost F1 Score:", xgb_f1_score)


joblib.dump(xgb_classifier, 'xgb_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')