import csv

def extractColumnsAsTuplesFromfile(filename):

    with open(filename,newline='') as csvfile:

       coordinates_List_Of_Tuples=[]

       spamreader=csv.reader(csvfile,delimiter=' ')

       for row in spamreader:

           coordinates_List_Of_Tuples.append((float(row[1]),float(row[2])))

       #coordinates_List_Of_Tuples=coordinates_List_Of_Tuples[0:10]

    return coordinates_List_Of_Tuples