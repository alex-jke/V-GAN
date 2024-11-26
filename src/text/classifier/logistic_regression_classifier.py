from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from classifier.classifier import Classifier
from dataset.dataset import Dataset


class LogisticRegressionClassifier(Classifier):

    def __init__(self, dataset: Dataset):
        self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        self.model = LogisticRegression(max_iter=200)
        self.x_train, self.y_train = dataset.get_training_data()
        self.x_test, self.y_test = dataset.get_testing_data()

    def train(self):
        # Transform the training data
        x_train_tfidf = self.tfidf.fit_transform(self.x_train)

        # Initialize and train the Logistic Regression model
        self.model.fit(x_train_tfidf, self.y_train)

    def predict(self):
        x_test_tfidf = self.tfidf.transform(self.x_test)
        y_pred = self.model.predict(x_test_tfidf)

        # Evaluate the model's performance
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

    def encode(self, data: str):
        sentence = [data]
        tfidf_sentence = self.tfidf.transform(sentence)
        return tfidf_sentence.toarray()

    def predict_single(self, data: str):
        encoded_data = self.encode(data)
        prediction = self.model.predict(encoded_data)
        return prediction
