
# 📧 Email Spam Detection using Machine Learning

Abstract : 

Email Spam has become a major problem nowadays, with Rapid growth of internet users, Email spams is also increasing. People are using them for illegal and unethical conducts, phishing and fraud. Sending malicious link through spam emails which can harm our system and can also seek in into your system. Creating a fake profile and email account is much easy for the spammers, they pretend like a genuine person in their spam emails, these spammers target those peoples who are not aware about these frauds. So, it is needed to Identify those spam mails which are fraud, this project will identify those spam by using techniques of machine learning, this project will discuss the machine learning algorithms and apply all these algorithm on our data sets and best algorithm is selected for the email spam detection having best precision and accuracy.


This project builds a spam detection system using natural language processing (NLP) and a logistic regression classifier. It analyzes email text messages and predicts whether they are **spam** or **ham** (not spam).

---


# FEATURES:
Text Preprocessing: Tokenization, stopword removal, and stemming
Feature Extraction: TF-IDF or Bag of Words
Machine Learning Models: Naïve Bayes, SVM, Random Forest, etc.
Performance Evaluation: Accuracy, Precision, Recall, and F1-score
Jupyter Notebook Implementation for easy experimentation


# DATASET:
The dataset used for training and testing is stored in mail_data.csv.
Contains labeled emails as Spam (1) or Not Spam (0).


# MODEL PERFORMANCE:
Achieves high accuracy in spam detection.
Suitable for real-world applications like email filtering systems.



## 🗂️ Dataset Description

- **Source**: SMS Spam Collection Dataset
- **File**: `mail_data.csv`
- **Structure**:
  - `Category`: Label (`spam` or `ham`)
  - `Message`: The actual email or SMS content

---

## ⚙️ Project Workflow

### 1. 📥 Data Loading
```python
df = pd.read_csv('mail_data.csv')
```
- Load the dataset using pandas.
- Replace null values with empty strings using `data = df.where((pd.notnull(df)), '')`.

### 2. 🧹 Data Preprocessing
```python
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1
```
- Convert categorical labels to binary:
  - `spam → 0`
  - `ham → 1`

### 3. ✂️ Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
```
- Split the data into training (80%) and testing (20%) sets.

### 4. 🧠 Feature Extraction (TF-IDF)
```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
```
- Convert text messages into numerical vectors using TF-IDF.
- Remove English stop words and lowercase all text.

### 5. 🏋️ Model Training
```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```
- Train a logistic regression model on the TF-IDF features.

### 6. 📊 Model Evaluation
```python
accuracy_score(Y_train, model.predict(X_train_features))
accuracy_score(Y_test, model.predict(X_test_features))
```
- Evaluate model performance using accuracy score:
  - **Training Accuracy**: ~96.77%
  - **Test Accuracy**: ~96.68%

### 7. 🔮 Spam Prediction Example
```python
input_your_mail = ["Congratulations! You've won a FREE iPhone 15! Click here to claim now: [scam-link]"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
```
- Predict whether a new message is spam or ham.
- Output:
  ```
  [1]
  Ham Mail
  ```

---

## 📦 Requirements

Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn
```

---

## 📁 Project Structure

```
Email-Spam-Detection/
│
├── mail_data.csv              # Dataset
├── spam_detection.py          # Main Python script
└── README.md                  # Project documentation
```

---

## 🚀 Future Enhancements

- ✅ Add confusion matrix and classification report
- ✅ Experiment with other models (SVM, Random Forest, Naive Bayes)
- ✅ Deploy as a web app using Flask or Streamlit
- ✅ Integrate with email clients for real-time filtering

---

## 👨‍💻 Author

**Subrat**  
A methodical coder with a passion for precision, template fidelity, and efficient problem-solving.

---

## 📜 License

This project is open-source and free to use for educational and research purposes.


