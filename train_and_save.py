{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ca941192-a250-46a4-b0c5-0d05cf7c608e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "from tensorflow.keras.preprocessing.text import one_hot\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import nltk\n",
    "import re\n",
    "from nltk.stem import PorterStemmer\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from wordcloud import WordCloud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "02cd018c-4a75-40db-84e8-1f9df089d9a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = pd.read_csv(\"train.txt\", header=None, sep=\";\", names=[\"Comment\", \"Emotion\"], encoding=\"utf-8\")\n",
    "train_data['length'] = [len(x) for x in train_data['Comment']]\n",
    "lb = LabelEncoder()\n",
    "train_data['Emotion'] = lb.fit_transform(train_data['Emotion'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "de55aaac-9a00-46dd-be54-f0adabe45db9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "15545095-39bf-4434-9ac4-61cc368cee4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = train_data.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e3c32b8b-a1ac-44fe-a072-77238778b857",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\ainia\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "nltk.download('stopwords')\n",
    "stopwords = set(nltk.corpus.stopwords.words('english'))\n",
    "def clean_text(text):\n",
    "    stemmer = PorterStemmer()\n",
    "    text = re.sub(\"[^a-zA-Z]\", \" \", text)\n",
    "    text = text.lower()\n",
    "    text = text.split()\n",
    "    text = [stemmer.stem(word) for word in text if word not in stopwords]\n",
    "    return \" \".join(text)\n",
    "\n",
    "df['cleaned_comment'] = df['Comment'].apply(clean_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1d16d628-c4ef-4441-b705-ad16a972eabe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "===== Multinomial Naive Bayes =====\n",
      "\n",
      "Accuracy using TF-IDF: 0.655\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.93      0.31      0.46       427\n",
      "           1       0.91      0.24      0.38       397\n",
      "           2       0.58      0.98      0.73      1021\n",
      "           3       1.00      0.03      0.06       296\n",
      "           4       0.70      0.91      0.79       946\n",
      "           5       1.00      0.01      0.02       113\n",
      "\n",
      "    accuracy                           0.66      3200\n",
      "   macro avg       0.85      0.41      0.41      3200\n",
      "weighted avg       0.76      0.66      0.58      3200\n",
      "\n",
      "\n",
      "===== Logistic Regression =====\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ainia\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Accuracy using TF-IDF: 0.829375\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.79      0.83       427\n",
      "           1       0.84      0.73      0.78       397\n",
      "           2       0.78      0.94      0.85      1021\n",
      "           3       0.80      0.49      0.61       296\n",
      "           4       0.88      0.92      0.90       946\n",
      "           5       0.77      0.45      0.57       113\n",
      "\n",
      "    accuracy                           0.83      3200\n",
      "   macro avg       0.82      0.72      0.76      3200\n",
      "weighted avg       0.83      0.83      0.82      3200\n",
      "\n",
      "\n",
      "===== Random Forest =====\n",
      "\n",
      "Accuracy using TF-IDF: 0.845625\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.80      0.85      0.82       427\n",
      "           1       0.83      0.83      0.83       397\n",
      "           2       0.83      0.90      0.86      1021\n",
      "           3       0.82      0.65      0.72       296\n",
      "           4       0.91      0.88      0.90       946\n",
      "           5       0.73      0.68      0.70       113\n",
      "\n",
      "    accuracy                           0.85      3200\n",
      "   macro avg       0.82      0.80      0.81      3200\n",
      "weighted avg       0.85      0.85      0.84      3200\n",
      "\n",
      "\n",
      "===== Support Vector Machine =====\n",
      "\n",
      "Accuracy using TF-IDF: 0.8190625\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.79      0.83       427\n",
      "           1       0.84      0.71      0.77       397\n",
      "           2       0.76      0.93      0.84      1021\n",
      "           3       0.81      0.45      0.58       296\n",
      "           4       0.88      0.91      0.89       946\n",
      "           5       0.79      0.47      0.59       113\n",
      "\n",
      "    accuracy                           0.82      3200\n",
      "   macro avg       0.82      0.71      0.75      3200\n",
      "weighted avg       0.82      0.82      0.81      3200\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ainia\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(df['cleaned_comment'],df['Emotion'],test_size=0.2,random_state=42)\n",
    "tfidf_vectorizer = TfidfVectorizer()\n",
    "X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)\n",
    "X_test_tfidf = tfidf_vectorizer.transform(X_test)\n",
    "classifiers = {\n",
    "    \"Multinomial Naive Bayes\": MultinomialNB(),\n",
    "    \"Logistic Regression\": LogisticRegression(),\n",
    "    \"Random Forest\": RandomForestClassifier(),\n",
    "    \"Support Vector Machine\": SVC(),\n",
    "}\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7ac4f738-ac73-4fbe-9051-263e19ea4ff2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "===== Multinomial Naive Bayes =====\n",
      "\n",
      "Accuracy using TF-IDF: 0.655\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.93      0.31      0.46       427\n",
      "           1       0.91      0.24      0.38       397\n",
      "           2       0.58      0.98      0.73      1021\n",
      "           3       1.00      0.03      0.06       296\n",
      "           4       0.70      0.91      0.79       946\n",
      "           5       1.00      0.01      0.02       113\n",
      "\n",
      "    accuracy                           0.66      3200\n",
      "   macro avg       0.85      0.41      0.41      3200\n",
      "weighted avg       0.76      0.66      0.58      3200\n",
      "\n",
      "\n",
      "===== Logistic Regression =====\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ainia\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Accuracy using TF-IDF: 0.829375\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.79      0.83       427\n",
      "           1       0.84      0.73      0.78       397\n",
      "           2       0.78      0.94      0.85      1021\n",
      "           3       0.80      0.49      0.61       296\n",
      "           4       0.88      0.92      0.90       946\n",
      "           5       0.77      0.45      0.57       113\n",
      "\n",
      "    accuracy                           0.83      3200\n",
      "   macro avg       0.82      0.72      0.76      3200\n",
      "weighted avg       0.83      0.83      0.82      3200\n",
      "\n",
      "\n",
      "===== Random Forest =====\n",
      "\n",
      "Accuracy using TF-IDF: 0.8471875\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.80      0.85      0.82       427\n",
      "           1       0.84      0.83      0.84       397\n",
      "           2       0.84      0.90      0.87      1021\n",
      "           3       0.81      0.62      0.70       296\n",
      "           4       0.91      0.89      0.90       946\n",
      "           5       0.74      0.69      0.71       113\n",
      "\n",
      "    accuracy                           0.85      3200\n",
      "   macro avg       0.82      0.80      0.81      3200\n",
      "weighted avg       0.85      0.85      0.85      3200\n",
      "\n",
      "\n",
      "===== Support Vector Machine =====\n",
      "\n",
      "Accuracy using TF-IDF: 0.8190625\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.79      0.83       427\n",
      "           1       0.84      0.71      0.77       397\n",
      "           2       0.76      0.93      0.84      1021\n",
      "           3       0.81      0.45      0.58       296\n",
      "           4       0.88      0.91      0.89       946\n",
      "           5       0.79      0.47      0.59       113\n",
      "\n",
      "    accuracy                           0.82      3200\n",
      "   macro avg       0.82      0.71      0.75      3200\n",
      "weighted avg       0.82      0.82      0.81      3200\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ainia\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "for name, clf in classifiers.items():\n",
    "    print(f\"\\n===== {name} =====\")\n",
    "    clf.fit(X_train_tfidf, y_train)\n",
    "    y_pred_tfidf = clf.predict(X_test_tfidf)\n",
    "    accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)\n",
    "    print(f\"\\nAccuracy using TF-IDF: {accuracy_tfidf}\")\n",
    "    print(\"Classification Report:\")\n",
    "    print(classification_report(y_test, y_pred_tfidf))\n",
    "\n",
    "lg = LogisticRegression()\n",
    "lg.fit(X_train_tfidf, y_train)\n",
    "lg_y_pred = lg.predict(X_test_tfidf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c502ad0a-a4da-4fcc-806a-f2e7d3242cdb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i didnt feel humiliated\n",
      "Prediction : sadness\n",
      "Label : 4\n",
      "================================================================\n",
      "i feel strong and good overall\n",
      "Prediction : joy\n",
      "Label : 2\n",
      "================================================================\n",
      "im grabbing a minute to post i feel greedy wrong\n",
      "Prediction : anger\n",
      "Label : 0\n",
      "================================================================\n",
      "He was speechles when he found out he was accepted to this new job\n",
      "Prediction : joy\n",
      "Label : 2\n",
      "================================================================\n",
      "This is outrageous, how can you talk like that?\n",
      "Prediction : anger\n",
      "Label : 0\n",
      "================================================================\n",
      "I feel like im all alone in this world\n",
      "Prediction : sadness\n",
      "Label : 4\n",
      "================================================================\n",
      "He is really sweet and caring\n",
      "Prediction : love\n",
      "Label : 3\n",
      "================================================================\n",
      "You made me very crazy\n",
      "Prediction : sadness\n",
      "Label : 4\n",
      "================================================================\n",
      "i am ever feeling nostalgic about the fireplace i will know that it is still on the property\n",
      "Prediction : love\n",
      "Label : 3\n",
      "================================================================\n",
      "i am feeling grouchy\n",
      "Prediction : anger\n",
      "Label : 0\n",
      "================================================================\n",
      "He hates you\n",
      "Prediction : anger\n",
      "Label : 0\n",
      "================================================================\n"
     ]
    }
   ],
   "source": [
    "def predict_emotion(input_text):\n",
    "    cleaned_text = clean_text(input_text)\n",
    "    input_vectorized = tfidf_vectorizer.transform([cleaned_text])\n",
    "\n",
    "    # Predict emotion\n",
    "    predicted_label = lg.predict(input_vectorized)[0]\n",
    "    predicted_emotion = lb.inverse_transform([predicted_label])[0]\n",
    "    label =  np.max(lg.predict(input_vectorized))\n",
    "\n",
    "    return predicted_emotion,label\n",
    "\n",
    "# Example usage \n",
    "sentences = [\n",
    "            \"i didnt feel humiliated\",\n",
    "            \"i feel strong and good overall\",\n",
    "            \"im grabbing a minute to post i feel greedy wrong\",\n",
    "            \"He was speechles when he found out he was accepted to this new job\",\n",
    "            \"This is outrageous, how can you talk like that?\",\n",
    "            \"I feel like im all alone in this world\",\n",
    "            \"He is really sweet and caring\",\n",
    "            \"You made me very crazy\",\n",
    "            \"i am ever feeling nostalgic about the fireplace i will know that it is still on the property\",\n",
    "            \"i am feeling grouchy\",\n",
    "            \"He hates you\"\n",
    "            ]\n",
    "for sentence in sentences:\n",
    "    print(sentence)\n",
    "    pred_emotion, label = predict_emotion(sentence)\n",
    "    print(\"Prediction :\",pred_emotion)\n",
    "    print(\"Label :\",label)\n",
    "    print(\"================================================================\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c6835c37-31d8-4537-a0d0-88e1d2aeb951",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "pickle.dump(lg,open(\"logistic_regresion.pkl\",'wb'))\n",
    "pickle.dump(lb,open(\"label_encoder.pkl\",'wb'))\n",
    "pickle.dump(tfidf_vectorizer,open(\"tfidf_vectorizer.pkl\",'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ce8d1bd6-5d92-4ed0-a9b9-92d1dc19e572",
   "metadata": {},
   "outputs": [],
   "source": [
    "def text_cleaning(df, column, vocab_size, max_len):\n",
    "    stemmer = PorterStemmer()\n",
    "    corpus = []\n",
    "\n",
    "    for text in df[column]:\n",
    "        text = re.sub(\"[^a-zA-Z]\", \" \", text)\n",
    "        text = text.lower()\n",
    "        text = text.split()\n",
    "        text = [stemmer.stem(word) for word in text if word not in stopwords]\n",
    "        text = \" \".join(text)\n",
    "        corpus.append(text)\n",
    "\n",
    "    one_hot_word = [one_hot(input_text=word, n=vocab_size) for word in corpus]\n",
    "    pad = pad_sequences(sequences=one_hot_word, maxlen=max_len, padding='pre')\n",
    "    return pad\n",
    "\n",
    "# Text cleaning and encoding\n",
    "x_train = text_cleaning(train_data, \"Comment\", vocab_size=11000, max_len=300)\n",
    "y_train = to_categorical(train_data[\"Emotion\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "aa96d976-b822-4656-8e6f-83a42aeeee0f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ainia\\anaconda3\\Lib\\site-packages\\keras\\src\\layers\\core\\embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m163s\u001b[0m 622ms/step - accuracy: 0.3188 - loss: 1.6857\n",
      "Epoch 2/10\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ainia\\anaconda3\\Lib\\site-packages\\keras\\src\\callbacks\\early_stopping.py:153: UserWarning: Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: accuracy,loss\n",
      "  current = self.get_monitor_value(logs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m147s\u001b[0m 586ms/step - accuracy: 0.5715 - loss: 1.2453\n",
      "Epoch 3/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m144s\u001b[0m 576ms/step - accuracy: 0.8102 - loss: 0.6067\n",
      "Epoch 4/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m147s\u001b[0m 588ms/step - accuracy: 0.8883 - loss: 0.3558\n",
      "Epoch 5/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m152s\u001b[0m 606ms/step - accuracy: 0.9134 - loss: 0.2660\n",
      "Epoch 6/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m160s\u001b[0m 638ms/step - accuracy: 0.9372 - loss: 0.1963\n",
      "Epoch 7/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m145s\u001b[0m 581ms/step - accuracy: 0.9497 - loss: 0.1572\n",
      "Epoch 8/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m207s\u001b[0m 600ms/step - accuracy: 0.9554 - loss: 0.1309\n",
      "Epoch 9/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m149s\u001b[0m 597ms/step - accuracy: 0.9632 - loss: 0.1166\n",
      "Epoch 10/10\n",
      "\u001b[1m250/250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m148s\u001b[0m 590ms/step - accuracy: 0.9684 - loss: 0.0964\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x11d0f3654f0>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = Sequential()\n",
    "model.add(Embedding(input_dim=11000, output_dim=150, input_length=300))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(LSTM(128))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(Dense(64, activation='sigmoid'))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(Dense(6, activation='softmax'))\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Train the model\n",
    "callback = EarlyStopping(monitor=\"val_loss\", patience=2, restore_best_weights=True)\n",
    "model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, callbacks=[callback])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "eafeb23d-5d15-4b41-ae14-4145e728c828",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i feel strong and good overall\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 811ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 160ms/step\n",
      "joy : 0.9987903237342834\n",
      "\n",
      "\n",
      "im grabbing a minute to post i feel greedy wrong\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 155ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 130ms/step\n",
      "anger : 0.9975230097770691\n",
      "\n",
      "\n",
      "He was speechles when he found out he was accepted to this new job\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 165ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 168ms/step\n",
      "sadness : 0.5538296103477478\n",
      "\n",
      "\n",
      "This is outrageous, how can you talk like that?\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 165ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 170ms/step\n",
      "anger : 0.9597368836402893\n",
      "\n",
      "\n",
      "I feel like im all alone in this world\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 158ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 165ms/step\n",
      "sadness : 0.9930880665779114\n",
      "\n",
      "\n",
      "He is really sweet and caring\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 149ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 167ms/step\n",
      "love : 0.8361496925354004\n",
      "\n",
      "\n",
      "You made me very crazy\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 150ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 157ms/step\n",
      "joy : 0.7405491471290588\n",
      "\n",
      "\n",
      "i am ever feeling nostalgic about the fireplace i will know that it is still on the property\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 140ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 171ms/step\n",
      "love : 0.9928304553031921\n",
      "\n",
      "\n",
      "i am feeling grouchy\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 163ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 154ms/step\n",
      "fear : 0.7366381287574768\n",
      "\n",
      "\n",
      "He hates you\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 139ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 155ms/step\n",
      "anger : 0.8587385416030884\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "def sentence_cleaning(sentence):\n",
    "    stemmer = PorterStemmer()\n",
    "    corpus = []\n",
    "    text = re.sub(\"[^a-zA-Z]\", \" \", sentence)\n",
    "    text = text.lower()\n",
    "    text = text.split()\n",
    "    text = [stemmer.stem(word) for word in text if word not in stopwords]\n",
    "    text = \" \".join(text)\n",
    "    corpus.append(text)\n",
    "    one_hot_word = [one_hot(input_text=word, n=11000) for word in corpus]\n",
    "    pad = pad_sequences(sequences=one_hot_word, maxlen=300, padding='pre')\n",
    "    return pad\n",
    "\n",
    "# load model and predict \n",
    "sentences = [\n",
    "            \"i feel strong and good overall\",\n",
    "            \"im grabbing a minute to post i feel greedy wrong\",\n",
    "            \"He was speechles when he found out he was accepted to this new job\",\n",
    "            \"This is outrageous, how can you talk like that?\",\n",
    "            \"I feel like im all alone in this world\",\n",
    "            \"He is really sweet and caring\",\n",
    "            \"You made me very crazy\",\n",
    "            \"i am ever feeling nostalgic about the fireplace i will know that it is still on the property\",\n",
    "            \"i am feeling grouchy\",\n",
    "            \"He hates you\"\n",
    "            ]\n",
    "for sentence in sentences:\n",
    "    print(sentence)\n",
    "    sentence = sentence_cleaning(sentence)\n",
    "    result = lb.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]\n",
    "    proba =  np.max(model.predict(sentence))\n",
    "    print(f\"{result} : {proba}\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "445d7e9e-76ca-4592-905a-66f8b55795b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "model.save('model1.h5')\n",
    "\n",
    "# Save the LabelEncoder\n",
    "with open('lb1.pkl', 'wb') as f:\n",
    "    pickle.dump(lb, f)\n",
    "\n",
    "# Save vocabulary size and max length\n",
    "vocab_info = {'vocab_size': 11000, 'max_len': 300}\n",
    "with open('vocab_info.pkl', 'wb') as f:\n",
    "    pickle.dump(vocab_info, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c15ff58e-6cb5-4dff-a639-4ff6bae12ec3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.18.0\n"
     ]
    }
   ],
   "source": [
    "import tensorflow\n",
    "print(tensorflow.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1113e472-c5c7-40a2-997a-720a08cc7ccb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
