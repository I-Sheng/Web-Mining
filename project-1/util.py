import sys
import math

#http://www.scipy.org/
try:
    from numpy import dot
    from numpy.linalg import norm
except:
    print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
    sys.exit()

def removeDuplicates(list):
    """ remove duplicates from a list """
    return set((item for item in list))

def euclidean(vector1, vector2):
    if sum(vector1) == 0 and sum(vector2) == 0:
        return float('inf')

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(vector1, vector2)))

def cosine(vector1, vector2):
    """ related documents j and q are in the concept space by comparing the vectors :
        cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
    # Add while the word in query is not in the corpus
    if sum(vector1) == 0:
        return 0

    if sum(vector2) == 0:
        return 0

    return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

