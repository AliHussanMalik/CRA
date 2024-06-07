# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tkinter as tk
from tkinter import messagebox

# Load the dataset
train_df = pd.read_csv('trainFYP.csv')

# Combine 'TITLE' and 'ABSTRACT' columns for text data
train_df['Text'] = train_df['TITLE'] + ' ' + train_df['ABSTRACT']

# EDA (Exploratory Data Analysis)
plt.figure(figsize=(12, 10))
categories = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

for i, category in enumerate(categories, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=category, data=train_df)
    plt.title(f'Distribution of {category}')

plt.tight_layout()
plt.show()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(train_df['Text'], train_df[categories], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train machine learning models for each category
models = {}
for category in categories:
    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X_train_tfidf, y_train[category])
    models[category] = model
    joblib.dump(model, f'{category}_model.joblib')

# Evaluate models on test data
for category, model in models.items():
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test[category], y_pred)
    print(f'{category} Accuracy: {accuracy}')
    print(classification_report(y_test[category], y_pred))

# Save the trained models for future use
joblib.dump(model, 'your_model_filename.joblib')

# GUI Interface using Tkinter
def predict_categories(text):
    predictions = {}
    for category, model in models.items():
        # TF-IDF Vectorization
        text_tfidf = vectorizer.transform([text])
        # Predict category
        predictions[category] = model.predict(text_tfidf)[0]
    return predictions

def on_predict_button_click():
    user_input = entry.get()
    predictions = predict_categories(user_input)
    result_text = "Predicted categories:\n"
    for category, prediction in predictions.items():
        result_text += f'{category}: {prediction}\n'
    messagebox.showinfo('Prediction Result', result_text)

# Create Tkinter window
window = tk.Tk()
window.title('Research Article Category Prediction')

# Create GUI components
label = tk.Label(window, text='Enter the research article text:')
entry = tk.Entry(window, width=50)
predict_button = tk.Button(window, text='Predict Categories', command=on_predict_button_click)

# Place GUI components in the window
label.pack(pady=10)
entry.pack(pady=10)
predict_button.pack(pady=10)

# Start Tkinter event loop
window.mainloop()