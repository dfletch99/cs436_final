import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#potential missing values for:
#	work_class
#	occupation
#	native_country
class DataEntry:
	def __init__(self, age, work_class, final_weight, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country):
		self.age = age
		self.work_class = work_class
		self.final_weight = final_weight
		self.education = education
		self.education_num = education_num
		self.marital_status = marital_status
		self.occupation = occupation
		self.relationship = relationship
		self.race = race
		self.sex = sex
		self.capital_gain = capital_gain
		self.capital_loss = capital_loss
		self.hours_per_week = hours_per_week
		self.native_country = native_country

	def toArray(self):
		return [
				self.age, self.work_class, self.final_weight,
				self.education, self.education_num, self.marital_status,
				self.occupation, self.relationship, self.race, self.sex,
				self.capital_gain, self.capital_loss, self.hours_per_week,
				self.native_country
				]


work_class_enum = [
			"Private", "Self-emp-not-inc", "Self-emp-inc",
			"Federal-gov", "Local-gov", "State-gov", "Without-pay",
			"Never-worked"
			]

education_enum = [
			"Bachelors", "Some-college", "11th",
			"HS-grad", "Prof-school", "Assoc-acdm",
			"Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
			"12th", "Masters", "1st-4th", "10th",
			"Doctorate", "5th-6th", "Preschool"
			]

marital_status_enum = [
				"Married-civ-spouse", "Divorced", "Never-married",
				"Separated", "Widowed", "Married-spouse-absent",
				"Married-AF-spouse"
				]

occupation_enum = [
			"Tech-support", "Craft-repair", "Other-service",
			"Sales", "Exec-managerial", "Prof-specialty",
			"Handlers-cleaners", "Machine-op-inspct",
			"Adm-clerical", "Farming-fishing", "Transport-moving",
			"Priv-house-serv", "Protective-serv", "Armed-Forces"
			]

relationship_enum = [
			"Wife", "Own-child", "Husband",
			"Not-in-family", "Other-relative",
			"Unmarried"
]

race_enum = [
			"White", "Asian-Pac-Islander",
			"Amer-Indian-Eskimo", "Other", "Black"
]

sex_enum = ["Female", "Male"]

native_country_enum = [
			"United-States", "Cambodia", "England",
			"Puerto-Rico", "Canada", "Germany",
			"Outlying-US(Guam-USVI-etc)", "India", "Japan",
			"Greece", "South", "China", "Cuba", "Iran", "Honduras",
			"Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
			"Mexico", "Portugal", "Ireland", "France",
			"Dominican-Republic", "Laos", "Ecuador", "Taiwan",
			"Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
			"Scotland", "Thailand", "Yugoslavia", "El-Salvador",
			"Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
]

def extract(ignore_missing_attribute_entries=True):
	TESTFILE = "./data/adult.data"
	#ALL_ENTRIES = []
	RAW_ENTRIES = []
	CLASSES = []
	fo = open(TESTFILE, "r", encoding="utf-8")
	for line in fo:
		row = line.split(', ')
		if row[14] == '?':
			print("here")
		age = int(row[0])
		try:
			work_class = work_class_enum.index(row[1])
		except ValueError:
			if ignore_missing_attribute_entries:
				continue
			work_class = -2
		final_weight = int(row[2])
		education = education_enum.index(row[3])
		education_num = int(row[4])
		marital_status = marital_status_enum.index(row[5])
		try:
			occupation = occupation_enum.index(row[6])
		except ValueError:
			if ignore_missing_attribute_entries:
				continue
			occupation = -2
		relationship = relationship_enum.index(row[7])
		race = race_enum.index(row[8])
		sex = sex_enum.index(row[9])
		capital_gain = int(row[10])
		capital_loss = int(row[11])
		hours_per_week = int(row[12])
		try:
			native_country = native_country_enum.index(row[13])
		except ValueError:
			if ignore_missing_attribute_entries:
				continue
			native_country = -2
		over50k = row[14][0:len(row[14])-1]
		if over50k == ">50K":
			over50k = 1
		else:
			over50k = -1
		entry = DataEntry(age, work_class, final_weight, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country)
		#ALL_ENTRIES.append(entry)
		RAW_ENTRIES.append(entry.toArray())
		CLASSES.append(over50k)
	#return ALL_ENTRIES, RAW_ENTRIES, CLASSES
	return RAW_ENTRIES, CLASSES

def findMissing(data):
	total_missing_points = 0
	entries_with_multiple_missing = 0
	for row in data:
		found_one = False
		keep_checking = True
		for point in row:
			if point == -1:
				total_missing_points += 1
				if found_one and keep_checking:
					entries_with_multiple_missing += 1
					keep_checking = False
				found_one = True
	print("Total missing attributes: " + str(total_missing_points))
	print("Lines with multiple missing: " + str(entries_with_multiple_missing))

def split(data_array, class_array, train_fraction):
	cols = [
		"age", "work_class", "final_weight",
		"education", "education_num", "marital_status",
		"occupation", "relationship", "race", "sex",
		"capital_gain", "capital_loss",
		"hours_per_week", "native_country"
	]

	X = pd.DataFrame(np.array(data_array), columns=cols)
	y = pd.DataFrame(class_array, columns=["over50k"])
	train_data, test_data, train_classes, test_classes = train_test_split(X, y, train_size=train_fraction, shuffle=False)
	return train_data, test_data, train_classes, test_classes

def main():
	array_entries, _ = extract(True)
	findMissing(array_entries)


if __name__ == "__main__":
	main()