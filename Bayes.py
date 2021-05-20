from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import process

IGNORE_INCOMPLETE_ENTRIES = False
TRAIN_FRACTION = 0.67
#choose between most_frequent, median, and mean
#only relevant if IGNORE_INCOMPLETE_ENTRIES == False
MISSING_STRATEGY = "most_frequent"

def gnb(train_data, train_classes, test_data, test_classes):
	transformer = SimpleImputer(missing_values=-2, strategy=MISSING_STRATEGY)
	classifier = make_pipeline(
						transformer,
						GaussianNB()
	)
	classifier.fit(train_data, train_classes.values.ravel())
	predictions = classifier.predict(test_data)
	num_correct = 0.0
	for i in range(len(test_classes)):
		if predictions[i] == test_classes.iat[i, 0]:
			num_correct += 1
	accuracy = num_correct / len(test_classes)
	print("Default GaussianNB classifier has accuracy " + str(accuracy))

	classifier = make_pipeline(
		transformer,
		GaussianNB(var_smoothing=0)
	)
	classifier.fit(train_data, train_classes.values.ravel())
	predictions = classifier.predict(test_data)
	num_correct = 0.0
	for i in range(len(test_classes)):
		if predictions[i] == test_classes.iat[i, 0]:
			num_correct += 1
	accuracy = num_correct / len(test_classes)
	print("GaussianNB classifier without var_smoothing has accuracy " + str(accuracy))

def cnb(train_data, train_classes, test_data, test_classes):
	transformer = SimpleImputer(missing_values=-2, strategy=MISSING_STRATEGY)
	transformer.fit(test_data, test_classes)
	classifier = make_pipeline(
		transformer,
		CategoricalNB()
	)
	classifier.fit(train_data, train_classes.values.ravel())
	predictions = classifier.predict(test_data)
	num_correct = 0.0
	for i in range(len(test_classes)):
		if predictions[i] == test_classes.iat[i, 0]:
			num_correct += 1
	accuracy = num_correct / len(test_classes)
	print("CategoricalNB classifier with no smoothing has accuracy " + str(accuracy))

	best_accuracy = -1
	best_laplace = -1

	for laplace in range(1, 100):
		#decimal_laplace = (laplace*1.0) / 100
		#print(decimal_laplace)
		transformer = SimpleImputer(missing_values=-2, strategy=MISSING_STRATEGY)
		classifier = make_pipeline(
			transformer,
			CategoricalNB(alpha=laplace)
		)
		classifier.fit(train_data, train_classes.values.ravel())
		predictions = classifier.predict(test_data)
		num_correct = 0.0
		for i in range(len(test_classes)):
			if predictions[i] == test_classes.iat[i, 0]:
				num_correct += 1
		accuracy = num_correct / len(test_classes)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_laplace = laplace
	print("Best classifier with smoothing has alpha = " + str(best_laplace))
	print("And accuracy " + str(best_accuracy))

def main():
	array_data, classifications = process.extract(IGNORE_INCOMPLETE_ENTRIES)
	train_data, test_data, train_classes, test_classes = process.split(array_data, classifications, TRAIN_FRACTION)
	gnb(train_data, train_classes, test_data, test_classes)
	cnb(train_data, train_classes, test_data, test_classes)


if __name__ == '__main__':
	main()