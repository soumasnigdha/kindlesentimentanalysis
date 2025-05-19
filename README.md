# Kindle Review Sentiment Analysis

This project focuses on building and evaluating different machine learning models for sentiment analysis on Kindle reviews. The goal is to classify reviews as positive or negative based on the review text and associated rating.

## Dataset

The project uses the `all_kindle_review.csv` dataset.

* **File**: `all_kindle_review.csv`

* **Content**: Contains Kindle review text (`reviewText`) and numerical ratings (`rating`).

* **Original Ratings**: 1 to 5 stars.

* **Sentiment Labeling**: Ratings are converted to a binary sentiment:

    * `0` (Negative): Ratings < 3

    * `1` (Positive): Ratings >= 3

* **Dataset Size (after labeling)**: 12,000 entries (8,000 positive, 4,000 negative).

## Preprocessing and Cleaning

The `reviewText` undergoes several cleaning and preprocessing steps to prepare it for analysis:

1.  **Lowercase Conversion**: Converts all text to lowercase.

2.  **Special Character Removal**: Removes non-alphanumeric characters, keeping spaces and hyphens.

3.  **Stopword Removal**: Eliminates common English stopwords.

4.  **HTML Tag Removal**: Removes HTML URLs and tags.

5.  **Extra Space Removal**: Replaces multiple spaces with single spaces.

6.  **Lemmatization**: Reduces words to their base or dictionary form using WordNetLemmatizer.

## Feature Engineering

Text data is transformed into numerical features using three different vectorization techniques:

1.  **Bag of Words (BoW)**: Using `CountVectorizer`.

2.  **TF-IDF (Term Frequency-Inverse Document Frequency)**: Using `TfidfVectorizer`.

3.  **Word Embeddings (Word2Vec)**:

    * A custom Word2Vec model is trained on the corpus.

    * Features are generated using:

        * Average Word2Vec vectors.

        * TF-IDF weighted average Word2Vec vectors.

## Model Training and Evaluation

The data is split into training (80%) and testing (20%) sets. The following machine learning models are trained and evaluated using each of the feature engineering methods:

* **Logistic Regression**

* **Random Forest Classifier**

* **Naive Bayes (MultinomialNB)**

Model performance is assessed using standard classification metrics:

* **Accuracy Score**

* **Classification Report** (Precision, Recall, F1-Score)

* **Confusion Matrix**

## Libraries and Dependencies

The project requires the following Python libraries:

* `pandas`

* `numpy`

* `nltk`

* `re`

* `bs4`

* `sklearn`

* `gensim`

## How to Run the Notebook

1.  **Clone the repository** (if applicable).

2.  **Obtain the dataset**: Make sure `all_kindle_review.csv` is in the same directory as the notebook.

3.  **Install dependencies**:

    ```
    pip install pandas numpy nltk scikit-learn beautifulsoup4 gensim

    ```

4.  **Download NLTK data**: Run the following in a Python environment or a notebook cell:

    ```
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')

    ```

5.  **Open the Jupyter Notebook**:

    ```
    jupyter notebook kindle_review_sentiment_analysis.ipynb

    ```

6.  **Run all cells**: Execute the notebook cells sequentially to perform the analysis, model training, and evaluation.
