from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
data = load_iris()
X = data.data
y = data.target

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LogisticRegression(max_iter=200)

# train
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# accuracy
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)