import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv("spam.csv")

x = dataframe["EmailText"]
y = dataframe["Label"]

x_train, y_train = x[0:4457], y[0:4457]
x_test, y_test = x[4457:], y[4457:]

cv = CountVectorizer()

training_data = cv.fit_transform(x_train)
testing_data = cv.transform(x_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

print(predictions)
print('Accuracy score:', format(accuracy_score(y_test, predictions)))
