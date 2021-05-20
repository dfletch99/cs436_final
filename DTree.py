from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import process
import copy

#if true, simply do not include entries with missing values
IGNORE_INCOMPLETE_ENTRIES = False
TRAIN_FRACTION = 0.67
#choose between most_frequent, median, and mean
#only relevant if IGNORE_INCOMPLETE_ENTRIES == False
MISSING_STRATEGY = "most_frequent"

def findBestTree(array_data, classifications):
	train_data, test_data, train_classes, test_classes = process.split(array_data, classifications, TRAIN_FRACTION)
	transformer = SimpleImputer(missing_values=-2, strategy=MISSING_STRATEGY)
	#transformer = transformer.fit(train_data, train_classes)

	best_accuracy = -1
	best_depth = -1
	best_criterion = -1
	best_splitter = -1
	best_impurity = -1
	impurity_values = [0, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3]

	best_classifier = []

	under_60_count = 0
	over_60_count = 0
	over_70_count = 0
	over_80_count = 0
	over_85_count = 0

	for depth in range(1, 11):
		print(depth)
		for criterion in ['gini', 'entropy']:
			for splitter in ['best', 'random']:
				for impurity in impurity_values:
					classifier = make_pipeline(
						transformer,
						DecisionTreeClassifier(
							max_depth=depth,
							criterion=criterion,
							splitter=splitter,
							min_impurity_decrease=impurity
						)
					)
					classifier.fit(train_data, train_classes)

					predictions = classifier.predict(test_data)
					num_correct = 0.0
					for i in range(len(test_classes)):
						if predictions[i] == test_classes.iat[i, 0]:
							num_correct += 1
					accuracy = num_correct / len(test_classes)

					if accuracy > 0.85:
						over_85_count += 1
					elif accuracy > 0.80:
						over_80_count += 1
					elif accuracy > 0.70:
						over_70_count += 1
					elif accuracy > 0.60:
						over_60_count += 1
					else:
						under_60_count += 1

					#don't use >= because that allows more complicated
					#models with the same accuracy
					if accuracy > best_accuracy:
						best_accuracy = accuracy
						best_depth = depth
						best_criterion = criterion
						best_splitter = splitter
						best_impurity = impurity
						best_classifier = copy.deepcopy(classifier)

	print("Best classifier has:")
	print("Depth " + str(best_depth))
	print("Criterion " + str(best_criterion))
	print("Split policy " + str(best_splitter))
	print("Minimum Impurity " + str(best_impurity))
	print("Best accuracy: " + str(best_accuracy))

	print("Classifiers with accuracy <= 60: " + str(under_60_count))
	print("Classifiers with accuracy > 60: " + str(over_60_count))
	print("Classifiers with accuracy > 70: " + str(over_70_count))
	print("Classifiers with accuracy > 80: " + str(over_80_count))
	print("Classifiers with accuracy > 85: " + str(over_85_count))
	return best_classifier

def main():
	array_data, classifications = process.extract(IGNORE_INCOMPLETE_ENTRIES)
	best_tree = findBestTree(array_data, classifications)


if __name__ == "__main__":
	main()