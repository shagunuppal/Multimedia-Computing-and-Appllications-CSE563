import argparse
import nltk
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from evaluation import evaluate_retrieval
from relevance_feedback import relevance_feedback, relevance_feedback_exp


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepFM.")
    parser.add_argument('--docs', nargs='?', default='data/med/MED.ALL',
                        help='Path of the document file.')
    parser.add_argument('--queries', nargs='?', default='data/med/MED.QRY',
                        help='Path of the query file.')
    parser.add_argument('--gt', nargs='?', default='data/med/MED.REL',
                        help='Path of the ground truth file.')
    parser.add_argument('--verbose', nargs='?', default=False,
                        help='Print additional information')

    return parser.parse_args()


def tokenize_text(docs):
    """
    custom tokenization function given a list of documents
    Parameters
        ----------
        docs : string
            a document

    Returns
    -------
    stems : list
        list of tokens
    """

    text = ''
    for d in docs:
        text += '' + d
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems


def load_docs(file):
    docs = []
    with open(file, 'r') as f:
        doc_split = f.read().replace('\n', ' ').replace('\r', ' ').split('.I')
    for l in doc_split[1:]:
        docs.append((''.join(re.sub(' +', ' ', l).split('.W')[1:])))
    return docs


def load_queries(file):
    queries = []
    with open(file, 'r') as f:
        q_split = f.read().replace('\n', ' ').replace('\r', ' ').split('.I')
    for l in q_split[1:]:
        queries.append( (''.join(re.sub(' +', ' ', l).split('.W')[1:])))
    return queries


def load_gt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return [(int(l.split()[0]), int(l.split()[2])) for l in lines]


def load_data(docs_path, query_path, gt_path):
    original_docs = load_docs(docs_path)
    queries = load_queries(query_path)
    gt = load_gt(gt_path)
    return original_docs, queries, gt


def tf_idf(docs, queries, tokenizer):
    """
    performs TF-IDF vectorization for documents and queries
    Parameters
        ----------
        docs : list
            list of documents
        queries : list
            list of queries
        tokenizer : custom tokenizer function

    Returns
    -------
    tfs : sparse array,
        tfidf vectors for documents. Each row corresponds to a document.
    tfs_query: sparse array,
        tfidf vectors for queries. Each row corresponds to a query.
    dictionary: list
        sorted dictionary
    """

    processed_docs = [d.lower().translate(string.punctuation) for d in docs]
    tfidf = TfidfVectorizer(stop_words='english', tokenizer=tokenizer)
    tfs = tfidf.fit_transform(processed_docs)
    tfs_query = tfidf.transform(queries)
    return tfs, tfs_query, tfidf


if __name__ == '__main__':
    args = parse_args()
    docs, queries, gt = load_data(args.docs, args.queries, args.gt)
    vec_docs, vec_queries, tfidf_model = tf_idf(docs, queries, tokenize_text)

    print ('\nBaseline Retrieval')
    sim_matrix = cosine_similarity(vec_docs, vec_queries)
    if args.verbose:
        for i in range(sim_matrix.shape[1]):
            ranked_documents = np.argsort(-sim_matrix[:, i])
            print ('Query:', i+1, 'Top relevant 10 documents:', ranked_documents[:10] + 1)
    evaluate_retrieval(sim_matrix, gt, verbose=args.verbose)

    print ('\nRetrieval with Relevance Feedback')
    rf_sim_matrix = relevance_feedback(vec_docs, vec_queries, sim_matrix)
    if args.verbose:
        for i in range(rf_sim_matrix.shape[1]):
            ranked_documents = np.argsort(-rf_sim_matrix[:, i])
            print ('Query:', i+1, 'Top relevant 10 documents:', ranked_documents[:10] + 1)
    evaluate_retrieval(rf_sim_matrix, gt, verbose=args.verbose)

    print ('\nRetrieval with Relevance Feedback and query expansion')
    rf_sim_matrix = relevance_feedback_exp(vec_docs, vec_queries, sim_matrix, tfidf_model)
    if args.verbose:
        for i in range(rf_sim_matrix.shape[1]):
            ranked_documents = np.argsort(-rf_sim_matrix[:, i])
            print ('Query:', i+1, 'Top relevant 10 documents:', ranked_documents[:10] + 1)
    evaluate_retrieval(rf_sim_matrix, gt, verbose=args.verbose)

