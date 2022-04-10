# This python file is use to create and populate the databases of all 6 classes 
#which helps the AI model to referes to the standard terms as a standard dictionary references.

# Thus this file we will create a databse and apply 
#CRUD operations on it in order to populate.
import sqlite3

nameOfDB = "dictionary.db"
tableList = [
			 "IdentityHate_Subject", 
			 "IdentityHate_Weak_Predicate",
			 "IdentityHate_Strong_Predicate",

  			 "Obscene_Subject",
  			 "Obscene_Weak_Predicate",
  			 "Obscene_Strong_Predicate",

  			 "Insult_Subject",
  			 "Insult_Weak_Predicate",
  			 "Insult_Strong_Predicate",

  			 "Threat_Subject",
  			 "Threat_Weak_Predicate",
  			 "Threat_Strong_Predicate",

  			 "Toxic_Subject",
  			 "Toxic_Weak_Predicate",
  			 "Toxic_Strong_Predicate",

  			 "SevereToxic_Subject",
  			 "SevereToxic_Weak_Predicate",
  			 "SevereToxic_Strong_Predicate"
  			  ]





#Creation of database file is as simple as making a connection
def createTable(nameOfDB, tableName, colName): #assuming only single column
	conn = makeDatabaseConnection(nameOfDB)
	cur = conn.cursor()  # A cursor object to execute the sql sqlQuery
	sqlQuery = f"""  CREATE TABLE IF NOT EXISTS {tableName} (id integer PRIMARY KEY AUTOINCREMENT,{colName} text  NOT NULL) """
	cur.execute(sqlQuery)
	return 

def insertContentIntoTable(nameOfDB, tableName,tableColumn ,content):
	conn = makeDatabaseConnection(nameOfDB)
	cur = conn.cursor()  # A cursor object to execute the sql sqlQuery
	sqlQuery = f""" INSERT INTO {tableName} ({tableColumn}) VALUES (?)"""
	# print(f"\n\n --*** The SQLQUERY IS : {sqlQuery}\n and the content is {content}\n")
	content = content.lower()
	cursor  = cur.execute(sqlQuery, (content,))
	conn.commit()
	print(f"{content} with the id : {cursor.lastrowid} is created succesfully!!")
	return



def makeDatabaseConnection(nameOfDB):
	conn = None
	try:
		conn = sqlite3.connect(nameOfDB)
	except sqlite3.error as e:
		print("SQLITE3 ERROR LOG " ,e)
	return conn




if __name__ == "__main__" :

	#CREATING THE REQUIRED DATABASE AND TABLES IN THAT DATABASE. 
	columnsList = {}
	for tableName in tableList:
		columnsList[tableName] = ''.join(tableName.split("_"))
		# print("COLUMNS : ",columnsList[tableName])
		createTable(nameOfDB, tableName, columnsList[tableName])

	#INSERTING CONTENT IN TABLES OF ABOVE DATABASES.
	contentOfTable = {}

	contentOfTable = { 
					 "IdentityHate_Subject": 
					 						{"IdentityHateSubject":["Christian","Christians","Hindu","Hindus","Muslim","Muslims","Indian","Indians","american","Americans", "Bible", "Quran", "Ramayan", "Gurugranth" , "Mahabharat" , "Mahabharatha"]}, 
		  			 "IdentityHate_Strong_Predicate": 	
		  			 						{"IdentityHateStrongPredicate":["bastard","motherfucker","motherfuckers","terrorist","terrorists","asshole","assholes","bitch","bitches","fucker","bullshit","bullshits"]},
		  			 "IdentityHate_Weak_Predicate": 	
		  			 						{"IdentityHateWeakPredicate":["bad","fake","dirty","cunt","dog","dogs","pig","fraud","dustbin", "dustbins","misleading"]},


		  			 "Obscene_Subject": 
		  			 						{"ObsceneSubject":["Squueze","Press","Bite", "Lick", "Suck", "Shake","Scratch" , "she", "red", "skirt", "figure", "man", "wish", "small", "mouth", "shut", "little" "angry", "piece", "watch", "child", "fun", "girlfriend", "park", "uncle", "lady", "mona", "lisa", "painting", "statue", "woman" ] },
		  			 "Obscene_Weak_Predicate": 
		  			 						{"ObsceneWeakPredicate":["melons","orange","milkpouch" , "lemon" , "lemons" , "oranges", "banana" , "pennis" , "gourd", "carrot","turns", "sleep", "grapes", "meet", "hug", "bushes", "hot"]},
		  			 "Obscene_Strong_Predicate": 
		  			 						{"ObsceneStrongPredicate":["boob","boobs","dick", "pussy" , "vagina", "fuck", "horny", "sexy", "boobs", "pussy", "shit", "porn", "dick", "kiss", "fuck", "fucking", "laid", "suck" ] },


		  			 "Toxic_Subject": 
		  			 						{"ToxicSubject":["i", "he", "she", "him", "people", "man", "me",  "dog", "my", "childhood",  "animals", "girl", "her", "minister",  "car", "farmers", "folks", "beware", "you" ] },
		  			 "Toxic_Strong_Predicate": 
		  			 						{"ToxicStrongPredicate":[ "stones", "blood", "wings", "head",  "torture", "devil", "monster", "innocent", "shame",  "piss", "pussy", "burn", "hounds", "beware", "fuck", "bite", "ass" ] },
		  			 "Toxic_Weak_Predicate": 
		  			 						{"ToxicWeakPredicate":["punched", "punch", "fist", "bald",  "old", "bus", "buy", "car",  "bump", "slap", "slapped", "follow", "throw", "childhood", "cut", "birds" , "little", "powerful", "electric", "heater", "beware"]
},


		  			 "SevereToxic_Subject": 
		  			 						{"SevereToxicSubject":["i", "will", "lion", "he", "her",  "body", "part", "man"]},
		  			 "SevereToxic_Strong_Predicate": 
		  			 						{"SevereToxicStrongPredicate":["cut", "threat", "insert", "iron", "rod", "vagina", "rapist", "bottle", "intestine",  "neck", "fall", "cage", "died" "split"]},
		  			 "SevereToxic_Weak_Predicate": 
		  			 						{"SevereToxicWeakPredicate":[]},


		  			 "Threat_Subject": 
		  			 						{"ThreatSubject":["blood","dick","boobs", "boob" ,  "name","dare", "trouble" , "man" , "blind" , "idiot" , "give""money",	"study"	, "worry" , "lower", "music" , "volume"]},
		  			 "Threat_Strong_Predicate": 
		  			 						{"ThreatStrongPredicate":["blow", "knock","jaw" ,"teach", "lesson", "viral","video", "police", "fail", "exam"]},
		  			 "Threat_Weak_Predicate": 
		  			 						{"ThreatWeakPredicate":["shed","cut","remove", "murder" , "murdered" , "bloody" , "motherfucker", "fucking" , "rape" , "murder" , "whore"]},


		  			 "Insult_Subject": 
		  			 						{"InsultSubject":["Blacks","Black","Whites", "White", "bihari", "shoes", "son", "parents", "money", "class",  "captain", "yourself", "brown", "men", "taller", "muslims", "muslim" ] },
		  			 "Insult_Weak_Predicate": 
		  			 						{"InsultWeakPredicate":["Fools","Animals","fools", "animal" , "animals" ,"stop", "eating", "eat", "hell", "bunk", "real", "permission", "go", 
"enter", "country", "girls", "coconut", "trees", "water", "know" ]  },
		  			 "Insult_Strong_Predicate": 
		  			 						{"InsultStrongPredicate":[ "pissed", "piss", "bitch", "bastard", "dumbass", "fuck", "theif", "bastards", "sex", "dwarf", "shorter",  "dickhead", "kissing", "kiss", "sister" , "fucker" ] }
					}

	count=0
	for tableName in tableList :
		# count+=1
		# if(count==4):
		# 	break
		CONTENTS = contentOfTable[tableName][ columnsList[tableName] ]
		print("\n")
		for content in CONTENTS:
			# print(f"\n\nTableName : {tableName}")
			# print(f"ColummnName : { columnsList[tableName] }")
			# print(f"Content : {content}")
			# print(f"WHOLE CONTENT : {CONTENTS}")
			columnName = columnsList[tableName]
			insertContentIntoTable(nameOfDB, tableName, columnName , content)




 





