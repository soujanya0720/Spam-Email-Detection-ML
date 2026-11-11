# Spam Email Detection using Machine Learning

A beginner-friendly Python project that classifies an email message as **Spam** or **Ham (Not Spam)**.

## Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Streamlit

## Project structure

```text
Spam_Email_Detection_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── spam_model.joblib       # created after training
└── data/
    └── emails.csv
```

## Setup in VS Code

1. Install Python 3.10 or newer.
2. Open this project folder in VS Code.
3. Open Terminal → New Terminal.
4. (Optional but recommended) Create a virtual environment:

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. Install the required libraries:
   ```bash
   python -m pip install -r requirements.txt
   ```

6. Train the model:
   ```bash
   python train_model.py
   ```

7. Start the web application:
   ```bash
   streamlit run app.py
   ```

8. Streamlit will show a local URL (usually http://localhost:8501). Open it in your browser, paste an email, and click **Check Email**.

## How it works

1. The dataset contains example email text and labels (`spam` or `ham`).
2. TF-IDF converts words into numerical features.
3. Multinomial Naive Bayes learns patterns from the labelled examples.
4. The app uses the trained model to predict the class of a new message.

## Important note

This is a starter demonstration using a small sample dataset, not a production-ready spam filter. Accuracy printed during training is only a basic evaluation and may vary. For a stronger internship submission, expand the dataset with a reputable labelled email dataset, remove duplicates, evaluate precision/recall/F1-score, and document the dataset source.

## Suggested demo video

Show:
1. The project folder and files in VS Code.
2. The training command and evaluation output.
3. The Streamlit app running.
4. One example predicted as spam and one example predicted as ham.
5. Briefly explain TF-IDF and Naive Bayes in your own words.
