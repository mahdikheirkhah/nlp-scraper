# 🕵️ NLP News Scraper & Risk Intelligence Platform

## 🎯 Project Goal
The goal of this platform is to transform raw news data into actionable **Risk Intelligence**. In a world of infinite information, manual monitoring is impossible. This project automates the detection of **Organizations (ORGs)**, classifies articles into **Topics**, analyzes **Sentiment**, and applies a specialized **Scandal Detection** logic to identify corporate environmental or financial disasters.

The platform is designed as a modular pipeline where a Scraper populates a data lake, and an NLP Engine enriches that data with deep semantic insights.

---

## 🚀 Execution Steps

### 1. Database Setup: Snowflake ❄️
We use **Snowflake** as our centralized SQL warehouse. This provides a professional, scalable environment to store both the raw scraped text and the final enriched results.
* **Storage:** We store `UUID`, `URL`, `Date`, `Headline`, and `Body`.
* **Security:** Connection is managed via RSA Private Key authentication for CI/CD safety.

### 2. Data Acquisition: RSS Scraper 📰
The `scraper_news.py` script targets **BBC** and **Fox News** RSS feeds. 
* **Volume:** We target a minimum of **300 articles** from the past week.
* **Parsing:** Using `BeautifulSoup` and `Requests`, the script extracts the full article body, ensuring the NLP engine has enough context for deep analysis.

That is a very solid choice. **LinearSVC (Support Vector Machine)** is often superior to Naive Bayes for text classification because it handles high-dimensional data (like TF-IDF vectors) and overlapping features (ngrams) much better.

Using `ngram_range=(1, 2)` is the "secret sauce" here—it allows the model to understand phrases like "stock market" or "data breach" as single concepts rather than just individual words.

Here is how you should document the **Audit/Training** section in your `README.md` to explain this specific technical choice:

---

### 🧠 Model Training & Performance (Audit Ready)

To meet the requirement of **>95% accuracy**, we implemented a **Linear Support Vector Classification (LinearSVC)** model. 

#### **Why LinearSVC?**
* **High-Dimensional Efficiency:** Text data processed via TF-IDF creates thousands of features. SVMs are mathematically designed to find the "optimal hyperplane" that separates these categories with the maximum margin.
* **N-Gram Integration:** By setting `ngram_range=(1, 2)`, the model captures both unigrams and bigrams, allowing it to distinguish between "Apple" (the fruit) and "Apple Inc" (the company) based on surrounding context.
* **Regularization:** We used `C=0.1` to prevent overfitting, ensuring the model generalizes well to new, unseen RSS feed data from BBC and Fox News.

#### **Training Pipeline**
```python
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', LinearSVC(C=0.1, random_state=42))
])
```

#### **Verification**
The auditor can verify the model's integrity by checking `results/learning_curves.png`. 
- **Training Score vs. Test Score:** A narrow gap between these two lines proves the model has high predictive power without "memorizing" the training data (overfitting).
- **Final Test Accuracy:** The console output during `training_model.py` execution will confirm the final percentage.

### 4. The NLP Enrichment Pipeline ⚙️
The `nlp_enriched_news.py` script is the core of the project. It processes the articles through the following stages:

#### **A. Text Preprocessing & Normalization**
Every article is cleaned by:
1.  **Lowercasing** and removing special punctuation.
2.  **Tokenization:** Breaking text into individual words and sentences.
3.  **Stop-word Removal:** Removing non-informative words (e.g., "the", "is").
4.  **Stemming:** Reducing words to their root form (e.g., "investigating" -> "investig").

#### **B. Entity Detection (NER)**
We use **spaCy's `en_core_web_md`** model to identify `ORG` entities. We specifically filter out "noise entities" (like 'The Government' or 'Congress') to focus strictly on corporations and specific organizations.

#### **C. Sentiment Analysis**
We utilize the **NLTK VADER** (Valence Aware Dictionary and sEntiment Reasoner). 
* **Why VADER?** It is a gold-standard pre-trained model specifically tuned for social media and news sentiment, providing a `compound` score from -1 (Extremely Negative) to 1 (Extremely Positive).

#### **D. Scandal Detection (The Logic)**
This is the most advanced part of the pipeline. It identifies if a company is involved in an environmental or financial disaster.

* **Embeddings Chosen:** We use **spaCy Medium Word Vectors (GloVe)**. 
    * *Why?* Unlike Bag-of-Words, embeddings capture the **semantic meaning**. For example, the vector for "Pollution" is mathematically close to "Contamination," even if the words share no letters.
* **Distance/Similarity:** We use **Cosine Similarity**.
    * *Why?* Cosine similarity measures the *angle* between two vectors. It is effective for text because it focuses on the orientation (theme) rather than the magnitude (word count).
* **The "Double-Lock" Logic:** An article is only flagged as a **Scandal** if it meets two criteria:
    1.  **Scandal Distance (Score > 0.85):** The sentence must be semantically very close to our defined "Risk Themes."
    2.  **Sentiment Gate (Score < -0.45):** The specific sentence mentioning the company must be negative. This prevents "False Positives" (e.g., a company announcing a new "Green Initiative" won't be flagged as a pollution scandal).

---

## 📊 Outputs & Results
* **`enhanced_news.csv`**: A full export of all 300+ articles with 10 columns of metadata.
* **Top 10 Flags**: The system automatically identifies and highlights the top 10 highest-risk articles based on the `Scandal_distance`.
* **Visualizations**: `results/` contains plots for topic distribution and sentiment trends across the scraped week.

---

## 🛠️ Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   ```

2. **Scrape Data:**
   ```bash
   python scripts/scraper_news.py
   ```

3. **Train Topic Model:**
   ```bash
   python results/training_model.py
   ```

4. **Run NLP Enrichment:**
   ```bash
   python scripts/nlp_enriched_news.py
   ```
