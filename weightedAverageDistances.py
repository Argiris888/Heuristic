
def ListOfWeightedDistances(listOfDistances,powerUsedToWeightedDistance):
       
       weightedListOfNN=[]
       
       for distance in listOfDistances :
       
           weightedDistance=(1/distance)**powerUsedToWeightedDistance
       
           weightedListOfNN.append(weightedDistance)
       
       return weightedListOfNN


