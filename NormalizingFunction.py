def NormalizingListOfWeightedDistances(weightedListOfNN):
    
    normalizedList=[]

    for wd in weightedListOfNN:

          normalizedElement=wd/sum(weightedListOfNN)

          normalizedList.append(normalizedElement)
    return normalizedList
     


 