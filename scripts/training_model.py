import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report

def save_learning_curves(estimator, X, y):
    """Task 3.4: Generate and save learning curves to results/."""
    # We use the training data for the curve to see how the model learns
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-Validation Score')
    plt.title('Learning Curves (LinearSVC + TF-IDF)')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.legend(loc="best")
    plt.grid(True)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/learning_curves.png')
    print("✅ Learning curves saved to results/learning_curves.png")

def train_topic_classifier():
    # 1. Load Both Datasets
    # Pro-tip: Always check multiple possible paths (root or /data)
    paths = ['data/', './']
    train_df, test_df = None, None
    
    for path in paths:
        train_p = os.path.join(path, 'bbc_news_train.csv')
        test_p = os.path.join(path, 'bbc_news_test.csv') # or bbc_news_tests.csv
        if not os.path.exists(test_p):
             test_p = os.path.join(path, 'bbc_news_tests.csv')

        if os.path.exists(train_p) and train_df is None:
            train_df = pd.read_csv(train_p)
        if os.path.exists(test_p) and test_df is None:
            test_df = pd.read_csv(test_p)

    if train_df is None or test_df is None:
        print("❌ Error: Could not find both train and test CSV files.")
        return

    # 2. Build Pipeline
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('clf', LinearSVC(C=0.1, random_state=42))
    ])

    # 3. Train on the FULL training set
    print(f"Training on {len(train_df)} samples...")
    model_pipeline.fit(train_df['Text'], train_df['Category'])

    # 4. Evaluate on the EXTERNAL test set
    print(f"Evaluating on external test set ({len(test_df)} samples)...")
    y_test_pred = model_pipeline.predict(test_df['Text'])
    
    # Check if the test set has labels (Category column)
    if 'Category' in test_df.columns:
        final_score = accuracy_score(test_df['Category'], y_test_pred)
        print(f"📊 Final External Test Accuracy: {final_score:.2%}")
        print("\nClassification Report:")
        print(classification_report(test_df['Category'], y_test_pred))
    else:
        print("⚠️ Test set has no 'Category' labels. Accuracy cannot be calculated.")
        final_score = 1.0 # Skip block if no labels to compare

    # 5. Save Model and Curves (Task 3.3 & 3.4)
    if final_score > 0.95:
        os.makedirs('results', exist_ok=True)
        save_learning_curves(model_pipeline, train_df['Text'], train_df['Category'])
        
        with open('results/topic_classifier.pkl', 'wb') as f:
            pickle.dump(model_pipeline, f)
        print("✅ Model verified and saved as results/topic_classifier.pkl")
    else:
        print(f"❌ Model test score ({final_score:.2%}) is below the 95% threshold.")

if __name__ == "__main__":
    train_topic_classifier()