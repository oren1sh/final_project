import sys
import os
from AnnManager import AnnManager
from PNPManager import PNPManager




def clrscr():
	"""
	Clearing the screen
	"""
	try:
		os.system('cls')  # For Windows
	except:
		os.system('clear')  # For Linux/OS X()
options = {
		1: "Neurons Net with train",
		2: "Neurons Net without train (Using pre trained model)",
		3: "Using SolvePNP"
	}
def netOption(train=False):
	netManager = NetManager()
	clrscr()
	print("Using Neurons Net")
	if(train):
		epoch = 2000
		batch = 20
		#while True:
		#	try:
		#		batch = int(input("Enter BATCH SIZE (enter 0 for the default 32) : \n {number of samples to work through before updating the internal model parameters}"))
		#		if batch <= 0:
		#			batch = 32
		#		break
		#	except:
		#		print("That's not a valid number!")
		#while True:
		#	try:
		#		epoch = int(input("Enter EPOCHS SIZE (enter 0 for the default 5000) : \n {number times that the learning algorithm will work through the entire training dataset}"))
		#		if epoch <= 0:
		#			epoch = 5000
		#		break
		#	except:
		#		print("That's not a valid number!")
		netManager.train(BATCH_SIZE=batch,EPOCHS=epoch)#training the model
	#end if train
	#netManager.predictByModel()#start the prediction
def solvePNPOption():
	pnpSolver = SolvePNPManager()
	clrscr()
	print("Using SOLVE PNP")
	pnpSolver.predict()
def main():

	netOption(1)
	#global options
	#print("Choose option from below :")
	#for option in options:
	#	print(str(option) + ". " + str(options[option]))
	#chosedOption = int(input())
	#print("You chose " + options[chosedOption])
	#if chosedOption <= 2:#use the net option
	#	netOption(train = (chosedOption == 1))#1 -> train = true
	#if chosedOption == 3:#solve PNP chosed
	#	solvePNPOption()
if __name__ == "__main__":
	main()


