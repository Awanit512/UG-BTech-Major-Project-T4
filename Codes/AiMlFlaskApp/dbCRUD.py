
# This python file is use to fetch/read update delete content from the table of the already populated databases of all 6 classes 
import sqlite3
from databasePopulator import  makeDatabaseConnection, tableList, nameOfDB





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

def updateTableContent(nameOfDB,tableName, oldValue, newValue):
	conn = makeDatabaseConnection(nameOfDB)
	pass

def deleteTableContent(nameOfDB, tableName, contentToDelete ):
	conn = makeDatabaseConnection(nameOfDB)
	pass


def fetchContentFromTable(nameOfDB, tableName,colName=None, colIndex = 0):
	conn = makeDatabaseConnection(nameOfDB)
	cur = conn.cursor()
	sqlQuery = """ """
	if colName!=None:
		sqlQuery = f""" SELECT {colName} FROM {tableName} """
	else:
		sqlQuery = f""" SELECT * FROM {tableName} """

	requiredTableContent = cur.execute(sqlQuery)
	if colIndex!=None:
		return [ row[colIndex] for row in requiredTableContent.fetchall()]
	else:
		return requiredTableContent




if __name__ == "__main__" : 

	columnsList = {}
	for tableName in tableList:
		columnsList[tableName] = ''.join(tableName.split("_"))
	
	requiredTableContent = fetchContentFromTable(nameOfDB , tableList[0], colName = columnsList[tableList[0]], colIndex = 0)
	print(requiredTableContent)



