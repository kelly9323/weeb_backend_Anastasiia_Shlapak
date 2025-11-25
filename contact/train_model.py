import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def load_dataset():
    df = pd.read_csv(
        'contact/data/twitter_training.csv',
        encoding='latin-1',
        header=None,
        names=['id', 'topic', 'sentiment', 'text']
    )
    return df

def prepare_data(df):
    """Prépare les données pour l'entraînement"""
    # Filtrer les lignes avec des valeurs manquantes
    df = df.dropna(subset=['text', 'sentiment'])
    
    # Mapper les sentiments : Positive = 1 (satisfait), autres = 0 (non satisfait)
    df['satisfaction'] = df['sentiment'].apply(
        lambda x: 1 if x == 'Positive' else 0
    )
    
    # Séparer les features (texte) et la target (satisfaction)
    X = df['text'].astype(str)
    y = df['satisfaction']
    
    return X, y

def get_models():
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )

    models = {
        "Logistic Regression": Pipeline([
            ('tfidf', vectorizer),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ]),
         "Decision Tree": Pipeline([
            ('tfidf', vectorizer),
            ('classifier', DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',
                max_depth=50,
                min_samples_split=10,
                min_samples_leaf=5
            ))
        ]),
        "Random Forest": Pipeline([
            ('tfidf', vectorizer),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                max_depth=50,
                min_samples_split=10,
                min_samples_leaf=5
            ))
        ])
    }

    return models


def train_model():
    df = load_dataset()
    print("Préparation des données")
    X, y = prepare_data(df)

    print(f"Nombre d'exemples: {len(X)}")
    print(f"Distribution des classes: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()
    best_model_name = None
    best_model = None
    best_accuracy = 0.0

    for name, model in models.items():
        print(f"\n--- Entraînement: {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Précision ({name}): {accuracy:.4f}")
        print("Rapport de classification:")
        print(classification_report(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    if best_model:
        model_path = os.path.join(
            os.path.dirname(__file__),
            'satisfaction_model.joblib'
        )
        joblib.dump(best_model, model_path)
        print(f"\nModèle sauvegardé ({best_model_name}) dans: {model_path}")

    return best_model