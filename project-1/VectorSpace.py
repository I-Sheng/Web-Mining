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
import pandas as pd

data_path = os.path.join(os.getcwd(), ".venv")
nltk.data.path.append(data_path)
# nltk.download('brown', download_dir=data_path)
# nltk.download('punkt_tab', download_dir=data_path)
# nltk.download('averaged_percept', download_dir=data_path)
# nltk.download('averaged_perceptron_tagger_eng', download_dir=data_path)
nltk.download('punkt', download_dir=data_path)
nltk.download('averaged_perceptron_tagger', download_dir=data_path)
print('\n\n')


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
    id_name :dict = {}

    #Tidies terms
    parser=None


    def __init__(self, documents, id_name, lang = "English"):
        self.documentVectors=[]
        self.parser = Parser()
        self.lang = lang
        self.id_name = id_name
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

    def search_tf(self, searchList, distance_matrix):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        ratings = [distance_matrix(queryVector, documentVector) for documentVector in self.documentVectors]
        return ratings

    def search_tf_cosine(self,searchList, verbose = True):
        if verbose:
            print("TF Weighting (Raw TF in course PPT) + Cosine Similarity")

        return self.search_tf(searchList, util.cosine)

    def search_tf_euclidean(self,searchList, verbose = True):
        if verbose:
            print("TF Weighting (Raw TF in course PPT) + Euclidean Distance")

        return self.search_tf(searchList, util.euclidean)

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


    def relevant_search_tf(self, searchList, distance_matrix):
        queryVector = self.buildQueryVector(searchList)
        ratings = [distance_matrix(queryVector, documentVector) for documentVector in self.documentVectors]

        if distance_matrix.__name__.endswith("cosine"):
            i = np.argsort(ratings)[-1:][0]
        else:
            i = np.argsort(ratings)[:1][0]

        queryVector = [a + b for a, b in zip(queryVector, self.feedback_query(i))]
        return [distance_matrix(queryVector, documentVector) for documentVector in self.documentVectors]

    def relevant_search_tf_cosine(self, searchList, verbose = True):
        if verbose:
            print("TF Weighting (Raw TF in course PPT) + Cosine Similarity")
        return self.relevant_search_tf(searchList, util.cosine)


    def relevant_search_tf_euclidean(self,searchList,verbose = True):
        if verbose:
            print("TF Weighting (Raw TF in course PPT) + Euclidean Distance")
        return self.relevant_search_tf(searchList, util.euclidean)

    def search_tfidf(self, searchList, distance_matrix):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryVector_tfidf = self.vector_to_tfidf(queryVector)
        return [distance_matrix(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]

    def search_tfidf_cosine(self,searchList, verbose = True):
        if verbose:
            print("TF-IDF Weighting (Raw TF in course PPT) + Cosine Similarity")

        return self.search_tfidf(searchList, util.cosine)


    def search_tfidf_euclidean(self,searchList, verbose = True):
        if verbose:
            print("TF-IDF Weighting (Raw TF in course PPT) + Euclidean Distance")

        return self.search_tfidf(searchList, util.euclidean)

    def print_top_ten(self, arr, matrix: str):
        if matrix == 'cosine':
            top_10_indices = np.argsort(arr)[-10:][::-1]
        else:
            top_10_indices = np.argsort(arr)[:10]

        for indice in top_10_indices:
            idx = int(indice)
            print(self.id_name[idx], arr[idx])

        print('-----------------------------------')

    def relevant_search_tfidf(self, searchList, distance_matrix):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryVector_tfidf = self.vector_to_tfidf(queryVector)
        ratings = [distance_matrix(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]


        if distance_matrix.__name__.endswith("cosine"):
            i = np.argsort(ratings)[-1:][0]
        else:
            i = np.argsort(ratings)[:1][0]
        queryVector = [a + b for a, b in zip(queryVector, self.feedback_query(i))]
        queryVector_tfidf = self.vector_to_tfidf(queryVector)

        return [distance_matrix(queryVector_tfidf, self.tfidf_emulator(i)) for i in range(len(self.documentVectors))]

    def relevant_search_tfidf_cosine(self,searchList, verbose = True):
        if verbose:
            print("TF-IDF Weighting (Raw TF in course PPT) + Cosine Similarity")
        return self.relevant_search_tfidf(searchList, util.cosine)

    def relevant_search_tfidf_euclidean(self,searchList, verbose = True):
        if verbose:
            print("TF-IDF Weighting (Raw TF in course PPT) + Euclidean Distance")
        return self.relevant_search_tfidf(searchList, util.euclidean)

    def MAP_10(self, queryList, answerList, matrix):
        if len(queryList) != len(answerList):
            print("Error: queryList isze is not equal to answerList")
            return

        def average_precision(top_10_indices, answer):
            hits: int = 0
            sum_precision = 0
            for i, idx in enumerate(top_10_indices, 1):
                if idx in answer:
                    hits += 1
                    sum_precision += hits / i
            return sum_precision / min(len(answer), 10); # This may improve
            # return sum_precision / 10

        size = len(queryList)
        total = 0
        def process_query(i):
            query = queryList[i]
            answer = answerList[i]
            arr = matrix(query, False)
            if matrix.__name__.endswith('cosine'):
                top_10_indices = np.argsort(arr)[-10:][::-1]
            else:
                top_10_indices = np.argsort(arr)[:10]

            return average_precision(top_10_indices, answer)

        total = sum(map(process_query, range(len(queryList))))
        map_score = total / size
        print(f'MAP@10:         {map_score:.6f}')

    def MRR_10(self, queryList, answerList, matrix):
        size = len(queryList)
        if size != len(answerList):
            print("Error: queryList isze is not equal to answerList")
            return

        def RR(top_10_indices, answer):
            for i in range(1, 11, 1):
                if top_10_indices[i-1] in answer:
                    return 1 / i
            return 0

        def process_query(i):
            query = queryList[i]
            answer = answerList[i]
            arr = matrix(query, False)
            if matrix.__name__.endswith('cosine'):
                top_10_indices = np.argsort(arr)[-10:][::-1]
            else:
                top_10_indices = np.argsort(arr)[:10]

            return RR(top_10_indices, answer)

        total = sum(map(process_query, range(len(queryList))))
        map_score = total / size
        print(f'MRR@10:         {map_score:.6f}')



    def RECALL_10(self, queryList, answerList, matrix):
        size = len(queryList)
        if size != len(answerList):
            print("Error: queryList isze is not equal to answerList")
            return

        def Recall(top_10_indices, answer):
            return len(set(top_10_indices) & set(answer)) / len(answer)

        def process_query(i):
            query = queryList[i]
            answer = answerList[i]
            arr = matrix(query, False)
            if matrix.__name__.endswith('cosine'):
                top_10_indices = np.argsort(arr)[-10:][::-1]
            else:
                top_10_indices = np.argsort(arr)[:10]

            return Recall(top_10_indices, answer)

        total = sum(map(process_query, range(len(queryList))))
        map_score = total / size
        print(f'RECALL@10:  {map_score:.6f}')




    def evaluate(self, queryList, answerList):

        print("Thank you for your patience, this might take a moment.")
        matrices = [self.search_tf_cosine, self.search_tfidf_cosine, self.search_tf_euclidean, self.search_tfidf_euclidean ]
        description = ['TF Cosine', 'TF-IDF Cosine', 'TF Euclidean', 'TF-IDF Euclidean']

        def evaluate_matrix(matrix):
            self.MAP_10(queryList, answerList, matrix)
            self.MRR_10(queryList, answerList, matrix)
            self.RECALL_10(queryList, answerList, matrix)

        for i in range(4):
            print(description[i])
            print('---------------------------')
            evaluate_matrix(matrices[i])
            print('---------------------------')

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
        vectorSpace = VectorSpace(documents, id_name)

        # problem. 1 - 1
        arr = vectorSpace.search_tf_cosine("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'cosine')

        # problem. 1 - 2
        arr = vectorSpace.search_tfidf_cosine("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'cosine')

        # problem. 1 - 3
        arr = vectorSpace.search_tf_euclidean("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'euclidean')

        # problem. 1 - 4
        arr = vectorSpace.search_tfidf_euclidean("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'euclidean')

        print('\n\n\n\n')
        print("Problem2:")
        # problem. 2 - 1
        arr = vectorSpace.relevant_search_tf_cosine("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'cosine')

        # problem. 2 - 2
        arr = vectorSpace.relevant_search_tfidf_cosine("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'cosine')

        # problem. 2 - 3
        arr = vectorSpace.relevant_search_tf_euclidean("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'euclidean')

        # problem. 2 - 4
        arr = vectorSpace.relevant_search_tfidf_euclidean("Typhoon Taiwan war")
        vectorSpace.print_top_ten(arr, 'euclidean')
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
        vectorSpace = VectorSpace(documents, id_name, lang = "Chinese")

        # problem. 3 - 1
        arr = vectorSpace.search_tf_cosine("資安 遊戲")
        vectorSpace.print_top_ten(arr, 'cosine')

        # problem. 3 - 2
        arr = vectorSpace.search_tfidf_cosine("資安 遊戲")
        vectorSpace.print_top_ten(arr, 'cosine')

    def p4():
        print("Problem4:")
        #test data
        documents: list  = []

        # directory_path = os.path.join(os.getcwd(), "EnglishNews")
        directory_path = os.path.join(os.getcwd(), "smaller_dataset", "collections")
        queries_path = os.path.join(os.getcwd(), "smaller_dataset", "queries")

        txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
        queries_files = glob.glob(os.path.join(queries_path, "*.txt"))


        df = pd.read_csv('./smaller_dataset/rel.tsv', sep='\t', header=None)

        # Convert second column to a list value
        df[1] = df[1].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df.columns = ['query', 'value']

        def query_to_answer(s:str):
            return df[df['query'] == s].iloc[0,1]

        i: int = 0
        id_name :dict = {}
        for file_path in txt_files:
            with open(file_path, 'r', encoding = 'utf-8') as file:
                content = file.read()
                documents.append(content)
                id_name[i] = os.path.basename(file_path)
                i += 1
            file.close()

        queryList = []
        answerList = []
        for query_path in queries_files:
            with open(query_path, 'r', encoding = 'utf-8') as file:
                content = file.read()
                queryList.append(content)
                query_name = os.path.basename(query_path).split('.')[0]
                answerList.append(query_to_answer(query_name))
            file.close()

        vectorSpace = VectorSpace(documents, id_name)
        vectorSpace.evaluate(queryList, answerList)

def main():
    Solutions.p12()
    Solutions.p3()
    Solutions.p4()

if __name__ == '__main__':
    main()
