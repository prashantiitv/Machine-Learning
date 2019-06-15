from sklearn import tree, svm
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

dtc = tree.DecisionTreeClassifier(random_state=1)
sgd = SGDClassifier(loss="hinge", random_state=1, penalty="l2", max_iter=25, tol=0.2)
svm = svm.SVC(kernel='linear')

# [height, weight, shoe_size]
X = [[176, 82, 45], [187, 73, 43], [161, 62, 39], [157, 51, 36], [167, 68, 41],
	 [191, 89, 48], [173, 65, 38], [176, 70, 41], [160, 55, 36], [170, 74, 43], [180, 81, 42]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Train them on the data
dtc = dtc.fit(X, Y)
sgd = sgd.fit(X, Y)
svm = svm.fit(X, Y)

predict_dtc = dtc.predict([[185, 68, 42]])
predict_sgd = sgd.predict([[185, 68, 42]])
predict_svm = svm.predict([[185, 68, 42]])

print('DecisionTreeClassifier says:')
print(predict_dtc)
print('GradientDecentClassifier says:')
print(predict_sgd)
print('SupportVectorMachine says:')
print(predict_svm)

# Print the best one!

