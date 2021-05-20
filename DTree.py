import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import process
import numpy as np
import copy

#if true, simply do not include entries with missing values
IGNORE_INCOMPLETE_ENTRIES = False
TRAIN_FRACTION = 0.67

def findBestTree(array_data, classifications):
	cols = [
		"age", "work_class", "final_weight",
		"education", "education_num", "marital_status",
		"occupation", "relationship", "race", "sex",
		"capital_gain", "capital_loss",
		"hours_per_week", "native_country"
	]

	X = pd.DataFrame(np.array(array_data), columns=cols)
	y = pd.DataFrame(classifications, columns=["over50k"])
	train_data, test_data, train_classes, test_classes = train_test_split(X, y, train_size=TRAIN_FRACTION, shuffle=False)
	'''
	transformer = FeatureUnion(
		transformer_list=[
			('features', SimpleImputer(missing_values=-2, strategy="median")),
			('indicators', MissingIndicator(missing_values=-2))
		]
	)
	'''
	transformer = SimpleImputer(missing_values=-2, strategy="mean")
	transformer = transformer.fit(test_data, test_classes)

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

	for depth in range(1, 16):
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
					#print("\t" + str(classifier.get_n_leaves()))

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
	class_data, array_data, classifications = process.extract(IGNORE_INCOMPLETE_ENTRIES)
	best_tree = findBestTree(array_data, classifications)


if __name__ == "__main__":
	main()