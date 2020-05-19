import numpy as np

#creating a list filled with the k nearest neighbours of my given point 
def FindKNearestNeighboursInThePath(my_distance_matrix,numberOfNearestNeighbours,givenPoint,myPath):
       
       #creating a list of distances for my given point
       my_distance_list=list(my_distance_matrix[givenPoint])

       #creating a list of  my k nearest neighbours of my given point
       kNearestNeighboursList=[]

       #creating a list of distances for my k nearest neighbours of my given point
       distanceKNNList=[]

       kNearestNeighbourIndex=givenPoint

       #to make sure that neighbours that are already in my path, are not included in my k nearest neighbours list of the given point
       for point in myPath:

                 my_distance_list[point]=float('inf')

       count=0

       #loop for finding the k nearest neighbours from my given point
       for k in range (numberOfNearestNeighbours):
              
               
               #in order to make infinite every point that is already in k nearest neighbours, and find next nearest neighbour
               my_distance_list[kNearestNeighbourIndex]=float('inf')
               
               minNextNearestNeighbourVisitDistance=min(my_distance_list)

               #we dont want to include in the kNearest neighbours infinite distances
               if minNextNearestNeighbourVisitDistance==float('inf'):

                   return  kNearestNeighboursList,distanceKNNList


               #finding the nearest neighbour each time
               kNearestNeighbourIndex=my_distance_list.index(minNextNearestNeighbourVisitDistance)

               #adding this neigbour to the list of k nearest neighbours of my present point
               kNearestNeighboursList.append(kNearestNeighbourIndex)

               #adding the distance of this  neighbour to my distance list of k nearest neighbours
               distanceKNNList.append(minNextNearestNeighbourVisitDistance)
               
               count+=1

               #returnin a tuple of 2 lists: the list of k nearest points from my given point, and a list of the distances between my given point and k nearest neighbours
       return  kNearestNeighboursList,distanceKNNList         




