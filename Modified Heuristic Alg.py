from scipy.spatial import distance_matrix

import ConstantsH
 
import csv

import numpy as np

from Plotting import MyPlot

from CSVreader import extractColumnsAsTuplesFromfile

from KNNeighbours import FindKNearestNeighboursInThePath

from weightedAverageDistances import ListOfWeightedDistances

from NormalizingFunction import NormalizingListOfWeightedDistances



#creating array from my given data using this function
coordinateListOfTuples=extractColumnsAsTuplesFromfile(filename='GreekCities.txt')

#creating a matrix filled with all the possible distances between my given points of visit       
mydistancematrix=np.array(distance_matrix(coordinateListOfTuples,coordinateListOfTuples))

#number of visits in order to visit each point and return to start point
numberOfVisits=len(coordinateListOfTuples)


#a list filled with the path(list of points) from each repetinion. Its a list of lists
listOfRepPaths=[]

#a list filled with the distances from each path for every repetition. Its a list of float numbers
listOfRepDistances=[]


#loops based on the number of loops that we are going to choose in order to select best path
for reps in range(ConstantsH.numberOfAlgorithmReps):

       #the starting and ending point of my path
       myNextPointOfVisit=0

       #the list of points of my path for every repetition
       myPath=[0]

       #the total distances for the path of each repetition
       totalDistanceOfPath=0


       # finding a path for each repetition, we use -1 because we dont include ending point
       # in each repetiion of this process we gonna find my next point of visit and its k nearest neighbours  until there is no more points for visit, except the starting point
       for i in range(numberOfVisits-1):

           #to make sure that we dont gonna have empty list of nearest neighbours
           if numberOfVisits!=len(myPath):
             
                    #using the function that returnes a tuple of 2 lists:t he list of k nearest points from my present point, and the list of the distances between my present point and k nearest neighbours
                    listOfNearestNeighbours,listOfDistances=FindKNearestNeighboursInThePath(mydistancematrix,ConstantsH.numberOfNearestNeighbours,myNextPointOfVisit,myPath)
                    
                    #using the function for returning a list of the weighted distances for my k nearest neighbours for my present point
                    weightedListOfNN=ListOfWeightedDistances(listOfDistances,ConstantsH.powerUsedToWeightedDistance)
                    
                    #normalizing the list of the weighted distances between my present point and its k nearest neighbours
                    normalizedList=NormalizingListOfWeightedDistances(weightedListOfNN)
                    
                    #randomly choosing my next point of visit from my list of my present's point k nearest neighbours, based on the probability of each neighbour
                    myNextPointOfVisit=np.random.choice(listOfNearestNeighbours, p=normalizedList) 
                    
                    NextPointOfVisitDistance=listOfDistances[list(listOfNearestNeighbours).index(myNextPointOfVisit)]
                    
                    #adding my next point of visit to my path list, for this repetition
                    myPath.append(myNextPointOfVisit)
                    
                    #calculating the total distance of my path for this repetition
                    totalDistanceOfPath+=NextPointOfVisitDistance

       #adding the final point to the path , that is the starting point
       myPath.append(myPath[0])
       
       #adding to the total distance of this repetition path, the distance between the last visited point and the starting point
       totalDistanceOfPath+=mydistancematrix[myNextPointOfVisit][myPath[0]]

       #from each repetition we have a complete path, so we add this path in a list which is finally filled with all the paths from my repititions
       listOfRepPaths.append(myPath)

       #a list filled with the distances of each path
       listOfRepDistances.append(totalDistanceOfPath)
       
       
           
           
#finding the minimum distance within my list of paths distances created by the upper set of repetitions
minimumDistance=min(listOfRepDistances)

#finding the path with the minimum distance created by the upper set of repetitions
indexOfPathWithMinimumDistance=list(listOfRepDistances).index(minimumDistance)

pathWithMinimumDistance=listOfRepPaths[indexOfPathWithMinimumDistance]



#print the results => the path with minimum distance and its distance, as well as the parameters used
print(f'After {ConstantsH.numberOfAlgorithmReps} repeats, using {ConstantsH.numberOfNearestNeighbours} nearest neighbours, and using power of weight: {ConstantsH.powerUsedToWeightedDistance},the\
 path with minimum distance is the following:\n\n\n{pathWithMinimumDistance} \
     \n\n\nThe total distance is: {round(minimumDistance,4)}')



#plotting the final path from this set of repetitions
coordinateListOfTuples = [coordinateListOfTuples[i] for i in pathWithMinimumDistance]

MyPlot(coordinateListOfTuples)



