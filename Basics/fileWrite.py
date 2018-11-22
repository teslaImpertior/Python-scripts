# Isaac Tesla 10/3/2018

string1 = '12345'

#file process, 1. open, 2. write/read 3. close.

#open file
path = 'd:/Scripts/Python3/test.txt'

#write to file
newString = open(path, 'a') # a = append, w = write over.
newString.write(string1)

#read file
readString = open(path, 'r')
fileString = readString.read()

#close file
newString.close()
readString.close()
