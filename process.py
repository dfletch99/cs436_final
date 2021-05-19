#potential missing values for:
#	work_class
#	occupation
#	native_country
class DataEntry:
	def __init__(self, age, work_class, final_weight, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, over50k):
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
		self.over50k = over50k


def extract():
	TESTFILE = "./data/adult.data"
	ALL_ENTRIES = []
	fo = open(TESTFILE, "r", encoding="utf-8")
	for line in fo:
		row = line.split(', ')
		if row[14] == '?':
			print("here")
		age = int(row[0])
		work_class = row[1]
		final_weight = int(row[2])
		education = row[3]
		education_num = int(row[4])
		marital_status = row[5]
		occupation = row[6]
		relationship = row[7]
		race = row[8]
		sex = row[9]
		capital_gain = int(row[10])
		capital_loss = int(row[11])
		hours_per_week = int(row[12])
		native_country = row[13]
		over50k = row[14][0:len(row[14])-1]
		ALL_ENTRIES.append(DataEntry(age, work_class, final_weight, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, over50k))
	return ALL_ENTRIES

def main():
	entries = extract()
	for i in entries:
		print(i.work_class)


if __name__ == "__main__":
	main()