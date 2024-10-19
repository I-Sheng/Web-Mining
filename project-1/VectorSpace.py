from Parser import Parser
import util
import os
import glob
import numpy as np
import math


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    # Number of the index
    indexNumber = []

    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]
        self.tfidf_emulator = self.create_tfidf_emulator()



    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1

        self.indexNumber = [0 for i in range(len(vectorIndex))]

        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            if vector[self.vectorKeywordIndex[word]] == 0:
                self.indexNumber[self.vectorKeywordIndex[word]] += 1

            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model

        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        termList: list = [term for term in termList if term in self.vectorKeywordIndex]
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings


    def search_tf_cosine(self,searchList):
        print("TF Weighting (Raw TF in course PPT) + Cosine Similarity")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings



    def search_tf_euclidean(self,searchList):
        print("TF Weighting (Raw TF in course PPT) + Euclidean Distance")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings


    def vector_to_tfidf(self, vector):
        def idf_for_word(index):
            return math.log(len(self.documentVectors) / (1 + self.indexNumber[index]))

        return [vector[i] * idf_for_word(i) for i in range(len(vector))]

    def create_tfidf_emulator(self):
        v = []

        for documentVector in self.documentVectors:
            v.append(self.vector_to_tfidf(documentVector))

        def tfidf_emulator(i):
            return v[i]

        return tfidf_emulator


    def search_tfidf_cosine(self,searchList):
        print("TF-IDF Weighting (Raw TF in course PPT) + Cosine Similarity")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryVector_tfidf = self.vector_to_tfidf(queryVector)

        ratings = [util.cosine(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]
        #ratings.sort(reverse=True)
        return ratings



    def search_tfidf_euclidean(self,searchList):
        print("TF-IDF Weighting (Raw TF in course PPT) + Euclidean Distance")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]
        #ratings.sort(reverse=True)
        return ratings



if __name__ == '__main__':



    #test data
    documents: list  = []

    directory_path = os.path.join(os.getcwd(), "EnglishNews")

    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

    i: int = 0
    id_name :dict = {}
    for file_path in txt_files:
        with open(file_path, 'r', encoding = 'utf-8') as file:
            content = file.read()
            documents.append(content)
            id_name[i] = os.path.basename(file_path)
            i += 1
        file.close()





    # print(f'len of documents: {len(documents)}')

    # documents = ["The cat in the hat disabled",
                 # "A cat is a fine pet ponies.",
                 # "Dogs and cats make good pets.",
                 # "I haven't got a hat."]

    # documents = documents[:10]
    vectorSpace = VectorSpace(documents)



    # problem. 1 - 1
    arr = vectorSpace.search_tf_cosine(["Typhoon", "Taiwan", "war"])
    top_10_indices = np.argsort(arr)[-10:][::-1]
    for indice in top_10_indices:
        idx = int(indice)
        print(id_name[idx], arr[idx])

    # problem. 1 - 2
    arr = vectorSpace.search_tfidf_cosine(["Typhoon", "Taiwan", "war"])
    top_10_indices = np.argsort(arr)[-10:][::-1]
    for indice in top_10_indices:
        idx = int(indice)
        print(id_name[idx], arr[idx])


    # problem. 1 - 3
    arr = vectorSpace.search_tf_euclidean(["Typhoon", "Taiwan", "war"])
    top_10_indices = np.argsort(arr)[-10:][::-1]
    for indice in top_10_indices:
        idx = int(indice)
        print(id_name[idx], arr[idx])

    # problem. 1 - 2
    arr = vectorSpace.search_tfidf_euclidean(["Typhoon", "Taiwan", "war"])
    top_10_indices = np.argsort(arr)[-10:][::-1]
    for indice in top_10_indices:
        idx = int(indice)
        print(id_name[idx], arr[idx])
###################################################
