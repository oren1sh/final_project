
import csv
import datetime
import os
import subprocess
class CSVModel:
	def __init__(self,methodName):
		self.methodName = methodName
		self.rows = []
		self.rows.append(["","file name","Rx","Ry","Rz","Tx","Ty","Tz"])

	def addRow(self,id,fileName,rx,ry,rz,tx,ty,tz):
		"""
		Adding row to the csv lines
		"""
		self.rows.append([str(id),str(fileName),str(rx),str(ry),str(rz),str(tx),str(ty),str(tz)])

	def writeToCSV(self):
		"""
		Creating and writing the CSV File 
		"""
		csvFileName = 'OutputsData\\Output_' + str(self.methodName) + '_' + 'OREN_SHALEV' + '.csv'
		try:
			with open(csvFileName, 'w',newline='') as file:
				writer = csv.writer(file)
				writer.writerows(self.rows)
			print("************** THE FILE " + csvFileName + " HAS CREATED  ***************")
			absPath = os.path.abspath(csvFileName)
			subprocess.Popen('explorer /select,"' + absPath + '"')#open the folder in explorer and select the file

		except:
			print("************** ERROR WHILE CREATING THE FILE " + csvFileName + " ***************")

