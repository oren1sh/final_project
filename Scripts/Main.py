import sys
import os
from AnnModel import AnnModel
from PNPModel import PNPModel


AnnModel = AnnModel()
PNPModel = PNPModel()


def clrscr():
	"""
	Clearing the screen
	"""
	try:
		os.system('cls')  # For Windows
	except:
		os.system('clear')  # For Linux/OS X()
options = {
		1: "Ann as SolvePNP",
		2: "Using DetectorModel and SolvePNP"
	}

def AnnModelPre():
	clrscr()
	print("Using Ann as solvePNP")
	AnnModel.predictByModel()#start the prediction

def AnnTrain():
	clrscr()
	print("Using Neurons Net")
	epoch = 2000
	batch = 15
	AnnModel.train(BATCH_SIZE=batch,EPOCHS=epoch)#training the model

def PNPModelSelect():

	clrscr()
	print("Using SOLVE PNP")
	PNPModel.predict()
def main():
	#from GeneratorOfData import GeneratorOfData
	#GeneratorOfData = GeneratorOfData()
	#GeneratorOfData.run()
	PNPModelSelect()
	#AnnTrain()

	#AnnModel = AnnModel()
	#global options
	#print("hi and welcome to oren shalev final project in computer vision")
	#print("please select one of these options below to see it run :" + "\n")
	#for option in options:
	#	print(str(option) + ". " + str(options[option]))
	#chosedOption = int(input())
	#print("\n" + "You chose " + options[chosedOption])
	#if chosedOption == 1:
	#	AnnOption()
	#elif chosedOption == 2:
	#	solvePNPOption()
	#else:
	#	print("end of run....")
	
if __name__ == "__main__":
	main()


