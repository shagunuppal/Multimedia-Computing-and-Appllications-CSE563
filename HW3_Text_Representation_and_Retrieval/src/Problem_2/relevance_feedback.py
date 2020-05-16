import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity
from scipy.linalg import norm

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    a, b, c = 1.0, 0.9, 0.5
    num_epochs = 3
    num_queries = vec_queries.shape[0]

    for epoch in range(num_epochs):
        query_vector = np.zeros((vec_queries.shape))

        for index in range(num_queries):
            prev_query = vec_queries[index, :]

            t_sim = sim[:, index]
            sort_index = np.argsort(t_sim)

            relevant_docs, irrelevant_docs = vec_docs[sort_index[-n:], :], vec_docs[sort_index[:n], :]
            relevant_mean, irrelevant_mean = mean_value(relevant_docs), mean_value(irrelevant_docs)

            next_query = (a/n) * prev_query + (b/n) * relevant_mean - (c/n) * irrelevant_mean
            query_vector[index, :] = next_query

        rf_sim = cosine_similarity(vec_docs, query_vector)
        vec_queries = query_vector
        sim = rf_sim

    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    a, b, c, num_synonyms = 0.1, 0.9, 1.4, 5

    num_epochs = 3
    num_queries = vec_queries.shape[0]

    # normalizing the documents
    # vec_docs = norm_2d(vec_docs)
    vec_docs = vec_docs / np.sum(vec_docs, axis=1)
    dictionary = np.dot(np.transpose(vec_docs), vec_docs)

    for epoch in range(3):
        query_vector = np.zeros((vec_queries.shape))

        for index in range(num_queries):
            prev_query = vec_queries[index, :]
            prev_query = vector_2d(prev_query)

            max_val, max_arg = np.max(prev_query), np.argmax(prev_query)
            # arr = dictionary[max_arg, :].reshape(-1)
            # print(arr.shape, max_arg)
            # arr = np.argsort(arr)
            # arr = arr[:, -num_synonyms:]

            # print(arr.shape)
            # synonyms = np.array(arr)[0]

            synonyms = np.array(np.argsort(dictionary[max_arg, :], axis=1)[:, -num_synonyms:])[0]
            
            for syn in synonyms:
                prev_query[:, syn] = max_val
            
            prev_query = vector_2d(prev_query)

            t_sim = sim[:, index]
            sort_index = np.argsort(t_sim)
            
            relevant_docs, irrelevant_docs = vec_docs[sort_index[-n:], :], vec_docs[sort_index[:n], :]
            relevant_mean, irrelevant_mean = mean_value(relevant_docs), mean_value(irrelevant_docs)

            next_query = (a/n) * prev_query + (b / n) * relevant_mean - (c / n) * irrelevant_mean
            query_vector[index, :] = next_query

        rf_sim = cosine_similarity(vec_docs, query_vector)
        vec_queries = query_vector
        sim = rf_sim
    
    return rf_sim

def mean_value(vector):
    return np.mean(vector, axis=0)

def vector_2d(vector):
    return vector.reshape(1, -1)

def norm_2d(matrix):
    for i in range(matrix.shape[0]):
        norm_val = 0.0
        for j in range(matrix.shape[1]):
            norm_val += matrix[i, j] ** 2
        norm_val = np.sqrt(norm_val)
        matrix[i, :] /= norm_val

    return matrix
