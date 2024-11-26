#!/bin/bash

# Set directories and files
WT2G_DIR="../data/WT2G"
COLLECTION_FILE="data/collection/collection.jsonl"
STEMMED_INDEX_DIR="indexes/stemmed"
UNSTEMMED_INDEX_DIR="indexes/unstemmed"
QUERY_FILE="../data/topics.401-440.txt"
QRELS_FILE="../data/qrels.401-440.txt"

# Function to generate collection.jsonl
generate_collection_jsonl() {
  echo "Converting WT2G files to JSONL format..."
  if [[ ! -f $COLLECTION_FILE ]]; then
    python3 codes/convert_wt2g_to_jsonl.py
  else
    echo "Collection file already exists: $COLLECTION_FILE"
  fi
}

# Function to create a stemmed index
create_stemmed_index() {
  echo "Building stemmed Lucene index..."
  if [[ ! -d $STEMMED_INDEX_DIR ]]; then
    python3 -m pyserini.index.lucene \
      --collection JsonCollection \
      --input data/collection \
      --index $STEMMED_INDEX_DIR \
      --generator DefaultLuceneDocumentGenerator \
      --threads 6 \
      --storePositions \
      --storeDocvectors \
      --storeRaw \
      --stemmer porter
  else
    echo "Stemmed index already exists: $STEMMED_INDEX_DIR"
  fi
}

# Function to create an unstemmed index
create_unstemmed_index() {
  echo "Building unstemmed Lucene index..."
  if [[ ! -d $UNSTEMMED_INDEX_DIR ]]; then
    python3 -m pyserini.index.lucene \
      --collection JsonCollection \
      --input data/collection \
      --index $UNSTEMMED_INDEX_DIR \
      --generator DefaultLuceneDocumentGenerator \
      --threads 6 \
      --storePositions \
      --storeDocvectors \
      --storeRaw \
      --stemmer none
  else
    echo "Unstemmed index already exists: $UNSTEMMED_INDEX_DIR"
  fi
}

# Function to generate runs
generate_runs() {
  methods=("bm25" "laplace" "jelinek_merver")
  for method in "${methods[@]}"; do
    for index_type in "stemmed" "unstemmed"; do
      run_file="runs/${method}-${index_type}.run"
      index_dir="indexes/${index_type}"
      echo "Running $method on $index_type index..."
      if [[ ! -f $run_file ]]; then
        python3 codes/main.py --method $method --output $run_file --query $QUERY_FILE --index $index_dir
      else
        echo "Run file already exists: $run_file"
      fi
    done
  done
}

# Function to evaluate runs
evaluate_runs() {
  echo "Evaluating all retrieval runs..."
  for method in "bm25" "laplace" "jelinek_merver"; do
    for index_type in "stemmed" "unstemmed"; do
      run_file="runs/${method}-${index_type}.run"
      echo "Evaluating $run_file..."
      if [[ -f $run_file ]]; then
        perl trec_eval.pl $QRELS_FILE $run_file
      else
        echo "Run file not found: $run_file"
      fi
    done
  done
}

# Main script
case $1 in
  "collection_jsonl")
    generate_collection_jsonl
    ;;
  "stemmed")
    generate_collection_jsonl
    create_stemmed_index
    ;;
  "unstemmed")
    generate_collection_jsonl
    create_unstemmed_index
    ;;
  "all")
    generate_collection_jsonl
    create_stemmed_index
    create_unstemmed_index
    generate_runs
    ;;
  "runs")
    generate_runs
    ;;
  "score")
    evaluate_runs
    ;;
  *)
    echo "Usage: $0 {collection_jsonl|stemmed|unstemmed|all|runs|score}"
    ;;
esac

