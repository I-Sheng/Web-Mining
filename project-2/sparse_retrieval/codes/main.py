import argparse
from search import *
from util import *
from searcher import bm25_searcher, laplace_language_model, jelinek_merver_language_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/collection", type=str)
    parser.add_argument("--query", default="../data/topics.401.txt", type=str)
    parser.add_argument("--method", default="bm25", type=str,choices=["bm25", "laplace", "jelinek_merver"])
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--output", default='runs/bm25.run', type=str)

    args = parser.parse_args()

    if args.method == "bm25":
        searcher = bm25_searcher(args.index)

    elif args.method == "laplace":
        searcher = laplace_language_model(args.index)

    elif args.method == "jelinek_merver":
        searcher = jelinek_merver_language_model(args.index)

    query = read_title(args.query)
    search(searcher, query, args)
