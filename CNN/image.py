# https://www.youtube.com/watch?v=IqfPGcNStE8&list=PLQVvvaa0QuDffXBfcH9ZJuvctJV3OtB8A&index=2
# Isaac Tesla

from PIL import Image #pillow library for images
import numpy as np #for matrices
import matplotlib.pyplot as plt
import time
import functools
#import sys # to point python system folder to the root directory of the files

#sys.path.insert(0, '/images/numbers')

def createExamples():
    path = 'numArEx.txt'
    numberArrayExamples = open(path,'a') #append to file when it creates it
    numbersWeHave = range(0,10) #how many examples of each number do we have
    versionsWeHave = range(1,10)
    for eachNum in numbersWeHave:
        for eachVer in versionsWeHave:
            #print (str(eachNum)+'.'+str(eachVer))
            imgFilePath = '/images/numbers/'+str(eachNum)+'.'+str(eachVer)+'.png'
            ei = Image.open(imgFilePath)  #ei = example image
            eiar = np.array(ei)  #eiar = example image array
            eiar1 = str(eiar.tolist()) # put information into a list

            lineToWrite = str(eachNum) + ': :' + eiar1 + '\n'
            numberArrayExamples.write(lineToWrite)
            print(lineToWrite)


# threshold to make all images black or white
def threshold(imageArray):
    balanceAr = [] #balance array to consider every value, at the end we average this array
    newAr = imageArray

    for eachRow in imageArray:
        for eachPixel in eachRow:
            #avgNum = functools.reduce(lambda x, y: x + y, eachPixel[:3])/len(eachPixel[:3])
            avgNum = (eachPixel[0]+eachPixel[1]+eachPixel[2])/3
            balanceAr.append(avgNum)
    balance = functools.reduce(lambda x, y: x+ y, balanceAr)/len(balanceAr)
    for eachRow in newAr:
        for eachPixel in eachRow:
            if functools.reduce(lambda x, y: x+ y, eachPixel[:3])/len(eachPixel[:3]) > balance: #if more than average, this is a lighter colour and needs to be turned to white
                eachPixel[0] = 255 #red
                eachPixel[1] = 255 #green
                eachPixel[2] = 255 #blue
                eachPixel[3] = 255 #alpha
            else: #make this pixel black
                eachPixel[0] = 0 #red
                eachPixel[1] = 0 #green
                eachPixel[2] = 0 #blue
                eachPixel[3] = 255 #alpha
    return newAr

i = Image.open('images/numbers/0.1.png')
iar = np.asarray(i)#, dtype='int64')
#iar.setflags(write=True)

i2 = Image.open('images/numbers/y0.4.png')
iar2 = np.array(i2)#, dtype='int64')
#iar2.setflags(write=True)

i3 = Image.open('images/numbers/y0.5.png')
iar3 = np.array(i3)#, dtype='int64')
#iar3.setflags(write=True)

threshold(iar3)



fig = plt.figure()
ax1 = plt.subplot2grid((8,6), (0,0), rowspan=4, colspan=3)
ax2 = plt.subplot2grid((8,6), (4,0), rowspan=4, colspan=3)
ax3 = plt.subplot2grid((8,6), (0,3), rowspan=4, colspan=3)

ax1.imshow(iar)
ax2.imshow(iar2)
ax3.imshow(iar3)
plt.show()

#i = Image.open('images/numbers/0.1.png')
#image_array = np.asarray(i)

#print(image_array)
#i.show()
#plt.imshow(image_array)
#plt.show()
