# 📰 Fake News Detector

A machine learning-based **Fake News Detection System** that analyzes news articles and predicts whether the given news is **Real or Fake**.

## 📌 Project Overview

The spread of fake news on social media and online platforms has become a major problem. This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify news articles as real or fake based on their textual content.

The system takes news content as input, processes the text, extracts important features, and uses a trained machine learning model to make the prediction.

## 🎯 Objectives

* Detect fake and misleading news automatically.
* Apply Natural Language Processing to news text.
* Train a machine learning classification model.
* Classify news as **Real** or **Fake**.
* Provide a simple and user-friendly way to test news content.

## 🛠️ Technologies Used

* **Python**
* **Machine Learning**
* **Natural Language Processing (NLP)**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **Matplotlib**
* **Seaborn**
* **Flask** *(if used in the project)*
* **HTML/CSS/JavaScript** *(if a web interface is included)*

## 📂 Project Structure

```text
fake-news-detector/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   └── trained model files
│
├── templates/
│   └── HTML files
│
├── static/
│   ├── CSS files
│   └── JavaScript files
│
├── app.py
├── requirements.txt
├── README.md
└── other project files
```

> The exact folder structure may vary depending on the files in the project.

## 🔄 How It Works

The project follows these main steps:

```text
News Article
     ↓
Text Preprocessing
     ↓
Data Cleaning
     ↓
Feature Extraction
     ↓
Machine Learning Model
     ↓
Prediction
     ↓
Real / Fake
```

### 1. Data Collection

The project uses datasets containing examples of **real and fake news articles**.

### 2. Data Preprocessing

The text data is cleaned before training. This may include:

* Converting text to lowercase
* Removing punctuation
* Removing unnecessary characters
* Removing stop words
* Tokenization
* Text normalization

### 3. Feature Extraction

The cleaned text is converted into numerical features that can be understood by a machine learning algorithm.

Common techniques include **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### 4. Model Training

A machine learning classification algorithm is trained using the processed news dataset.

Possible models include:

* Logistic Regression
* Naive Bayes
* Support Vector Machine
* Random Forest

### 5. Prediction

When a user enters news content, the system processes the text and predicts:

```text
REAL NEWS
```

or

```text
FAKE NEWS
```

## 📊 Dataset

The project contains two main datasets:

* `Fake.csv` — contains fake news articles.
* `True.csv` — contains real news articles.

The datasets are used to train and evaluate the machine learning model.

**Note:** The dataset files are relatively large, so GitHub may display a large-file warning. For production repositories, Git LFS or an external dataset link can be used.

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/aishuaishu45793-gif/fake-news-detector.git
```

### Step 2: Open the project

```bash
cd fake-news-detector
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

If the project uses Flask, run:

```bash
python app.py
```

Then open the local URL displayed in the terminal, commonly:

```text
http://127.0.0.1:5000/
```

## 🧪 Example

### Input

```text
Scientists announce a new discovery that could change the future of technology.
```

### Output

```text
Prediction: REAL
```

Another news article may produce:

```text
Prediction: FAKE
```

The prediction depends on the trained model and the features extracted from the input text.

## 📈 Model Evaluation

The machine learning model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics help measure how effectively the model distinguishes between real and fake news.

## 🌟 Features

* 📰 Fake news classification
* 🤖 Machine learning-based prediction
* 🔤 NLP text processing
* 📊 Dataset-based model training
* ⚡ Fast prediction
* 🖥️ User-friendly interface *(if implemented)*

## 🚀 Future Enhancements

* Improve model accuracy using advanced NLP techniques.
* Implement deep learning models such as LSTM or Transformers.
* Add support for multiple languages.
* Add news URL verification.
* Integrate real-time news sources.
* Add explainable AI to show why an article was classified as fake.
* Deploy the application online.
* Add a browser extension for real-time fake news detection.

## ⚠️ Disclaimer

This project is developed for **educational and research purposes**. The prediction should not be considered a definitive determination of whether a news article is true or false. Users should verify important information using reliable sources.

## 👩‍💻 Author

**Aishwarya**

GitHub:
https://github.com/aishuaishu45793-gif

## 📄 License

This project is intended for educational purposes. You may modify and use the code for learning and academic projects.
