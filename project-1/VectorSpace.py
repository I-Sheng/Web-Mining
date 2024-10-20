from Parser import Parser
import util
import os
import glob
import numpy as np
import math
import nltk
from nltk import word_tokenize
from nltk.tag import pos_tag
import jieba

data_path = os.path.join(os.getcwd(), ".venv")
nltk.data.path.append(data_path)
# nltk.download('brown', download_dir=data_path)
# nltk.download('punkt_tab', download_dir=data_path)
# nltk.download('averaged_percept', download_dir=data_path)
# nltk.download('averaged_perceptron_tagger_eng', download_dir=data_path)
nltk.download('punkt', download_dir=data_path)
nltk.download('averaged_perceptron_tagger', download_dir=data_path)




class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex={}

    # Number of the index
    indexNumber = []

    #Tidies terms
    parser=None


    def __init__(self, documents=[], lang = "English"):
        self.documentVectors=[]
        self.parser = Parser()
        self.lang = lang
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document, True) for document in documents]
        self.tfidf_emulator = self.create_tfidf_emulator()



    def buildVector(self, doc):
        return self.makeVector(doc, True)

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string
        vocabularyString = " ".join(documentList)

        if self.lang == "English":
            vocabularyList = self.parser.tokenise(vocabularyString)
            #Remove common words which have no search value
            vocabularyList = self.parser.removeStopWords(vocabularyList)
            uniqueVocabularyList = util.removeDuplicates(vocabularyList)
        else:
            words = jieba.lcut(vocabularyString)
            uniqueVocabularyList = util.removeDuplicates(words)


        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1

        self.indexNumber = [0 for i in range(len(vectorIndex))]

        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString, create: bool, lang = "English"):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        if self.lang == "English":
            wordList = self.parser.tokenise(wordString)
            wordList = self.parser.removeStopWords(wordList)
        else:
            wordList = jieba.lcut(wordString)

        for word in wordList:
            if word not in self.vectorKeywordIndex:
                print("An error occur")
                print("word: ", word)
                print("wordString", wordString)
                return
            if create and vector[self.vectorKeywordIndex[word]] == 0:
                self.indexNumber[self.vectorKeywordIndex[word]] += 1

            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model


        return vector


    def buildQueryVector(self, string):
        """ convert query string into a term vector """

        if self.lang == "English":
            termList = string.split(" ")
        else:
            termList = jieba.lcut(string)

        termList: list = [term for term in termList if term.lower() in self.vectorKeywordIndex]
        query = self.makeVector(" ".join(termList), False)

        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
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

    def search_tf_cosine(self,searchList):
        print("TF Weighting (Raw TF in course PPT) + Cosine Similarity")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings

    def feedback_query(self, idx):

        def tag_single_word(word):
            tokens = word_tokenize(word)
            pos_tagged = pos_tag(tokens)
            return pos_tagged[0][1]

        v2 = self.documentVectors[idx]
        for w, i in self.vectorKeywordIndex.items():
            if v2[i] == 0:
                continue
            tag = tag_single_word(w)
            if tag == 'VB' or tag == 'NN':
                v2[i] = 0
            else:
                v2[i] /= 2

        return v2





    def relevant_search_tf_cosine(self, searchList):
        print("TF Weighting (Raw TF in course PPT) + Cosine Similarity")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        i = np.argsort(ratings)[-1:][0]
        queryVector = [a + b for a, b in zip(queryVector, self.feedback_query(i))]
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


    def relevant_search_tf_euclidean(self,searchList):
        print("TF Weighting (Raw TF in course PPT) + Euclidean Distance")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]

        i = np.argsort(ratings)[:1][0]
        queryVector = [a + b for a, b in zip(queryVector, self.feedback_query(i))]
        ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]

        return ratings


    def search_tfidf_cosine(self,searchList):
        print("TF-IDF Weighting (Raw TF in course PPT) + Cosine Similarity")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryVector_tfidf = self.vector_to_tfidf(queryVector)

        ratings = [util.cosine(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]

        return ratings


    def relevant_search_tfidf_cosine(self,searchList):
        print("TF-IDF Weighting (Raw TF in course PPT) + Cosine Similarity")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryVector_tfidf = self.vector_to_tfidf(queryVector)

        ratings = [util.cosine(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]

        i = np.argsort(ratings)[-1:][0]
        queryVector = [a + b for a, b in zip(queryVector, self.feedback_query(i))]
        queryVector_tfidf = self.vector_to_tfidf(queryVector)
        ratings = [util.cosine(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]

        return ratings

    def search_tfidf_euclidean(self,searchList):
        print("TF-IDF Weighting (Raw TF in course PPT) + Euclidean Distance")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryVector_tfidf = self.vector_to_tfidf(queryVector)

        ratings = [util.euclidean(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]
        #ratings.sort(reverse=True)
        return ratings

    def relevant_search_tfidf_euclidean(self,searchList):
        print("TF-IDF Weighting (Raw TF in course PPT) + Euclidean Distance")
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryVector_tfidf = self.vector_to_tfidf(queryVector)

        ratings = [util.euclidean(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]

        i = np.argsort(ratings)[0]
        queryVector = [a + b for a, b in zip(queryVector, self.feedback_query(i))]
        queryVector_tfidf = self.vector_to_tfidf(queryVector)
        ratings = [util.euclidean(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]
        #ratings.sort(reverse=True)
        return ratings


    def MAP_10(self, queryList, answerList):
        if len(queryList) != len(answerList):
            print("Here is error, qyeryList size is not equal to answerList")
            return

        def element_answer(top_10_indices, answer):
            return len(set(top_10_indices) & set(answer))

        size: int = len(queryList)
        total = 0
        top_10_list = []
        for i in range(size):
            query = queryList[i]
            answer = answerList[i]
            arr = vectorSpace.search_tfidf_cosine(query)
            top_10_indices = np.argsort(arr)[-10:][::-1]
            top_10_list.append(top_10_indices)

        total = sum(list(map(element_in_answer, top_10_list, answerList)))

        return total / size

class Solutions():
    def p12():
        print("Problem1:")
        #test data
        documents: list  = []

        # directory_path = os.path.join(os.getcwd(), "EnglishNews")
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

        # documents = documents[:10]
        vectorSpace = VectorSpace(documents)

        # problem. 1 - 1
        arr = vectorSpace.search_tf_cosine("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[-10:][::-1]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])

        # problem. 1 - 2
        arr = vectorSpace.search_tfidf_cosine("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[-10:][::-1]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])


        # problem. 1 - 3
        arr = vectorSpace.search_tf_euclidean("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[:10]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])

        # problem. 1 - 4
        arr = vectorSpace.search_tfidf_euclidean("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[:10]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])
        print('\n\n\n\n')

        print("Problem2:")
        # problem. 2 - 1
        arr = vectorSpace.relevant_search_tf_cosine("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[-10:][::-1]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])

        # problem. 2 - 2
        arr = vectorSpace.relevant_search_tfidf_cosine("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[-10:][::-1]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])


        # problem. 2 - 3
        arr = vectorSpace.relevant_search_tf_euclidean("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[:10]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])

        # problem. 2 - 4
        arr = vectorSpace.relevant_search_tfidf_euclidean("Typhoon Taiwan war")
        top_10_indices = np.argsort(arr)[:10]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])
        print('\n\n\n\n')

    def p3():
        print("Problem3:")
        #test data
        documents: list  = []

        # directory_path = os.path.join(os.getcwd(), "EnglishNews")
        directory_path = os.path.join(os.getcwd(), "ChineseNews")

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

        # documents = documents[:10]
        vectorSpace = VectorSpace(documents, lang = "Chinese")

        # problem. 3 - 1
        arr = vectorSpace.search_tf_cosine("資安 遊戲")
        top_10_indices = np.argsort(arr)[-10:][::-1]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])

        # problem. 3 - 2
        arr = vectorSpace.search_tfidf_cosine("資安 遊戲")
        top_10_indices = np.argsort(arr)[-10:][::-1]
        for indice in top_10_indices:
            idx = int(indice)
            print(id_name[idx], arr[idx])
        print("\n\n\n")

    def p4():
        print("Problem4:")
        #test data
        documents: list  = []

        # directory_path = os.path.join(os.getcwd(), "EnglishNews")
        directory_path = os.path.join(os.getcwd(), "smaller_dataset", "collections")
        queries_path = os.path.join(os.getcwd(), "smaller_dataset", "queries")

        txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
        queries_files = glob.glob(os.path.join(queries_path, "*.txt"))

        i: int = 0
        id_name :dict = {}
        for file_path in txt_files:
            with open(file_path, 'r', encoding = 'utf-8') as file:
                content = file.read()
                documents.append(content)
                id_name[i] = os.path.basename(file_path)
                i += 1
            file.close()
        # query & answer 如何帶入MAP未完成
        # documents = documents[:10]
        vectorSpace = VectorSpace(documents)





if __name__ == '__main__':
    # Solutions.p12()
    # Solutions.p3()
    Solutions.p4()

