from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import process

#if true, simply do not include entries with missing values
IGNORE_INCOMPLETE_ENTRIES = True
TRAIN_FRACTION = 0.67
TRAINING_CYCLES = 25
#choose between most_frequent, median, and mean
#only relevant if IGNORE_INCOMPLETE_ENTRIES == False
MISSING_STRATEGY = "most_frequent"

def basicMLP(train_data, train_classes, test_data, test_classes):
	classifier = MLPClassifier(random_state=42)
	classifier.fit(train_data, train_classes.values.ravel())
	#predictions = classifier.predict(test_data)
	accuracy = classifier.score(test_data, test_classes)
	print("Total accuracy for default MLP: " + str(accuracy))

def parameterMLP(train_data, train_classes, test_data, test_classes):
	#PARAMETERS
	# modify the following as desired
	num_hidden_layers = 15
	num_nodes_per_layer = 10
	solver = 'adam'					#'lbfgs', 'sgd', or 'adam'
	learning_rate = 'constant'		#'constant', 'invscaling', or 'adaptive'
	early_stopping = False
	alpha = 0.7						#0.1, 0.32, 1, 3.16, 10, etc (can be any float)
	#PARAMETERS

	hidden_layer_sizes = np.full(num_hidden_layers, num_nodes_per_layer)
	classifier = MLPClassifier(
								alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,
								random_state=42, solver=solver, learning_rate=learning_rate,
								early_stopping=early_stopping
								)
	classifier.fit(train_data, train_classes.values.ravel())
	# predictions = classifier.predict(test_data)
	accuracy = classifier.score(test_data, test_classes)
	print("Accuracy for parametrized MLP: " + str(accuracy))

def MLPScaling(train_data, train_classes, test_data, test_classes):
	# PARAMETERS
	# modify the following as desired
	num_hidden_layers = 3
	num_nodes_per_layer = 15
	solver = 'adam'  # 'lbfgs', 'sgd', or 'adam'
	learning_rate = 'invscaling'  # 'constant', 'invscaling', or 'adaptive'
	early_stopping = False
	alpha = 0.7  # 0.1, 0.32, 1, 3.16, 10, etc (can be any float)
	max_iter = 500
	# PARAMETERS

	hidden_layer_sizes = np.full(num_hidden_layers, num_nodes_per_layer)
	scaler = StandardScaler()
	scaler.fit(train_data)
	classifier = make_pipeline(
		scaler,
		MLPClassifier(
						max_iter=max_iter, verbose=True, random_state=42,
						solver=solver, learning_rate=learning_rate, early_stopping=early_stopping,
						alpha=alpha, hidden_layer_sizes=hidden_layer_sizes
						)
	)
	classifier.fit(train_data, train_classes.values.ravel())
	accuracy = classifier.score(test_data, test_classes)
	print("Accuracy for MLP with parameters and feature scaling: " + str(accuracy))

def main():
	array_data, classifications = process.extract(IGNORE_INCOMPLETE_ENTRIES)
	train_data, test_data, train_classes, test_classes = process.split(array_data, classifications, TRAIN_FRACTION)
	#basicMLP(train_data, train_classes, test_data, test_classes)
	#parameterMLP(train_data, train_classes, test_data, test_classes)
	MLPScaling(train_data, train_classes, test_data, test_classes)


if __name__ == '__main__':
	main()