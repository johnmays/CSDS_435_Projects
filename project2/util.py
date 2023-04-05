"""
Utility functions for CSDS 435 Project 2.
"""
import numpy as np
import matplotlib.pyplot as plt
import re

purples = ["#0a0612", "#392249", "#482980", "#673ab7",
           "#7a52aa", "#9779bd", "#b59fd0", "#d3c5e3"]

stopwords = ['', "wasn't", 'there', 'been', 'they', 'until', 'yourself', 'that', 'themselves', 'into', 'where', 'he', 'both', "it's", 'too', 'under', 'no', 'in', 'the', 'me', 'which', 'do', 'i', 'own', 'their', 'him', "that'll", 'up', '&amp;', 'same', 'at', 'a', 'rt', "aren't", 'who', 'on', 'we', 'being', 'can', 'against', 'your', 'she', 'again', 'whom', 'will', 'as', 'did', 'has', "isn't", 'few', 'through', 'down', "should've", "you'll", 'myself', 'himself', 'such', 'am', 'tip', 'most', 'but', 'while', 'to', 'if', "weren't", "won't", 'us', 'and', 'how', 'be', 'each', 'some', 'may', 'says', 'about', 'for', 'any', 'only', 'are', 'his', 'with', 'by', 'here', "doesn't", 'all', 'then', 'of', 'hers', 'them', 'or', 'theirs', 'because', 'having', 'itself', 'what', 'yourselves', "shouldn't", 'its', 'were', 'during', "didn't", 'further', 'nor', 'out', 'had', "you'd", 'not', 'you', 'could', 'our', 'my', 'other', 'herself', 'after', 'her', 'those', 'why', 'have', "hasn't", "you're", 'once', 'doing', 'below', 'yours', "she's", '--', 'is', 'off', 'an', 'via', "hadn't", 'from', 'ours', 'so', "wouldn't", "you've", 'this', 'over', 'it', 'between', 'when', "haven't", 'more', 'before', "couldn't", 'than', 'very', 'these', 'ourselves', "don't", 'above', 'does', 'was', 'should', 'rt', 'would', 'w/', 'just', '\r\n']


def load_data(path: str, stopwords = stopwords): 
    print('Importing data...')
    # format file into a list of tweets:
    lines = []
    with open(path, newline='\n') as file:
        lines = file.readlines()
    # take last column, take out punctuation, links, and hashtag characters, make lowercase, split into list of words
    for i, line in enumerate(lines):
        line = line.split('|')[2] 
        formatted_line = re.sub(r'!|:|,|\.|\||#', '', re.sub(r'http\S+', '', line))
        words_in_a_line = formatted_line.lower().split(' ')
        lines[i] = remove_stopwords(words_in_a_line, stopwords)
    # vocab = np.unique(np.concatenate(lines).flat)
    vocab = np.unique(sum(lines, []))
    # creating data vector X:
    n = np.size(vocab)
    m = len(lines)
    X = np.zeros((m,n))
    # populating X:
    for i, line in enumerate(lines):
        for word in line:
            j = np.where(vocab==word)[0]
            X[i,j] += 1

    print('X has {} examples and {} features in BOW format...'.format(m,n))
    return X, np.array(vocab)


def remove_stopwords(words, stopwords):
    return list(filter(lambda word: word not in stopwords, words))


def generate_statistics(X,vocab):
    print("Dataset Statistics:")
    m,n = np.shape(X)
    num_words = int(np.sum(X))
    print("# Number of Tweets: {}".format(m))
    print("# Number of Words (Total): {}".format(num_words))
    print("# Number of Tokens: {}".format(n))
    print("# Average Number of Words per Tweet: {}".format(round((num_words/m),3)))
    print("# Top 10 Legal Tokens: {}".format(vocab[np.flip(np.argsort(np.sum(X, axis=0)))[0:10]]))


def distance1(x1,x2):
    """
    unordered, normalized edit distance
    """
    return np.sum(np.abs(x1-x2))/(np.sum(x1)+np.sum(x2))


def distance2(x1,x2):
    """
    Euclidean distance
    """
    return np.sqrt(np.sum(np.power((x1-x2), 2)))


def create_distance_matrix(X:np.ndarray, distance_type=1) -> np.ndarray:
    if distance_type not in (1,2):
        raise ValueError('distance type should either be 1 or 2 (as ints).')
    if distance_type == 1:
        distance_measure = distance1
    else:
        distance_measure = distance2
    m,n = np.shape(X)
    distance_matrix = np.zeros(shape=(m,m))
    for i in range(m):
        print('progress: {}'.format(str(i/m)), end='\r')
        for j in range(i):
            distance_matrix[i,j] = distance_measure(X[i],X[j])
            distance_matrix[j,i] = distance_matrix[i,j]
    print(f'done with distance metric {str(distance_type)}!')
    return distance_matrix


def create_distance_matrices(X):
    distance_matrix_1 = create_distance_matrix(X,distance_type=1)
    distance_matrix_2 = create_distance_matrix(X,distance_type=2)
    return distance_matrix_1, distance_matrix_2


def save_distance_matrices(distance_matrix_1, distance_matrix_2, save_type='npy'):
    if save_type not in ('npy', 'txt'):
        raise ValueError("save_type should either be 'npy' or 'txt'.")
    if save_type == 'npy':
        with open('distance_matrices.npy', 'wb') as file:
            np.save(file, distance_matrix_1)
            np.save(file, distance_matrix_2)
    else: # then txt:
        # writes to txt file with separating character "," and a newline between the two matrices
        with open('distance_matrix_1.txt', 'w') as file:
            for i in range(np.shape(distance_matrix_1)[0]):
                file.write(str(list(distance_matrix_1[i]))[1:-1])
        with open('distance_matrix_2.txt', 'w') as file:
            for i in range(np.shape(distance_matrix_2)[0]):
                file.write(str(list(distance_matrix_2[i]))[1:-1])


def load_distance_matrices():
    with open('distance_matrices.npy', 'rb') as file:
        distance_matrix_1 = np.load(file)
        distance_matrix_2 = np.load(file)
    return distance_matrix_1, distance_matrix_2


def unique_distances(distance_matrix):
    # takes only the unique distances (bottom triangle section with the diagonal) from the distance matrix and returns them as a list
    m,n = distance_matrix.shape
    if m!=n:
        raise ValueError('distance_matrix should be square.')
    distances = []
    for i in range(m):
        for j in range(i+1):
            distances.append(distance_matrix[i,j])
    return np.array(distances)


def plot_distance_distributions(distance_matrix_1, distance_matrix_2):
    plt.rcParams['figure.dpi'] = 150
    distances_1 = unique_distances(distance_matrix_1)
    distances_2 = unique_distances(distance_matrix_2)
    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(9,3)
    fig.tight_layout()
    axes[0].hist(distances_1, bins=40, color=purples[4])
    axes[0].set_title('Distribution of Distances under Measure 1\n')
    axes[0].set_ylabel('Num instances')
    axes[0].set_xlabel('Distance')

    axes[1].hist(distances_2, bins=40, color=purples[4])
    axes[1].set_title('Distribution of Distances under Measure 2\n')
    axes[1].set_xlabel('Distance')
    plt.show();


def cluster_size_distribution(labels):
    sizes = np.zeros(int(np.max(labels)+1))
    for label in labels:
        sizes[int(label)] += 1
    frequencies = []
    size_set = list(set(list(sizes)))
    for size in size_set:
        frequencies.append(np.sum((sizes==size)))
    return size_set, frequencies


def plot_cluster_size_distributions(sizes, frequencies, cluster_name, distance_type):
    plt.figure(figsize=(5,4), dpi=100)
    plt.stem(sizes, frequencies, basefmt=purples[6], markerfmt=purples[6], linefmt=purples[6])
    plt.title(f'Cluster Size Dist. for {cluster_name} w/ {distance_type}')
    plt.ylabel('# of clusters')
    plt.xlabel('size of cluster')
    plt.show();


def SSE(X, cluster_labels):
    '''
    Within-cluster sum of squares error
    '''
    m,n = np.shape(X)
    k = np.max(cluster_labels)

    clusters = np.arange(k+1, dtype=int)
    means = np.zeros(((k+1), n))
    # find mean for each cluster
    for cluster in clusters:
        if X[cluster_labels==cluster].size != 0:
            means[cluster] = np.average(X[cluster_labels==cluster], axis=0)
        else:
            pass #stays as zeros
    # iterate through examples and find squared distance from cluster means
    sum_error = 0.0
    for x, label in zip(X, cluster_labels):
        sum_error += np.sum(np.square(x-means[label]))
    return sum_error


def BSS(X, cluster_labels):
    '''
    Between-cluster sum of squares error
    '''
    cluster_labels = np.array(cluster_labels)
    k = np.max(cluster_labels)

    clusters = np.arange(k+1, dtype=int)
    total_sample_mean = np.average(X, axis=0)
    
    sum_error = 0.0
    for cluster in clusters:
        if X[cluster_labels==cluster].size != 0:
            cluster_mean = np.average(X[cluster_labels==cluster], axis=0)
            cluster_size = np.sum((cluster_labels==cluster))
        else:
            cluster_mean = 0.0
            cluster_size = 0.0
        sum_error += cluster_size*np.sum(np.square(cluster_mean-total_sample_mean))
    
    return sum_error


def print_cluster_topics(X, vocab, cluster_labels):
    cluster_labels = np.array(cluster_labels)
    k = np.max(cluster_labels)
    print('Top words in the clusters:')
    clusters = set(list(cluster_labels))
    cluster_sizes = []
    for cluster in clusters:
        cluster_size = np.sum((cluster_labels==cluster))
        cluster_sizes.append(cluster_size)

    cluster_sizes, clusters = (list(t) for t in zip(*sorted(zip(cluster_sizes, clusters), reverse=True)))
    for i in range(np.min((k,8))):
        print("Cluster {}:".format(clusters[i]))
        print("Size {}:".format(cluster_sizes[i]))
        print("{}".format(vocab[np.flip(np.argsort(np.sum(X[cluster_labels==clusters[i]], axis=0)))[0:10]]))


def visualize_clustering(D, cluster_labels, cluster_name, distance_type):
    '''
    Should take the distance matrix, D, sort it vertically, and print with plt.imshow().
    '''
    # sort D by clusters
    k = np.max(cluster_labels)
    clusters = np.arange(k+1, dtype=int)
    sorted_D = None
    for cluster in clusters:
        if np.sum((cluster_labels==cluster)) == 0: # no members in cluster
            pass
        elif sorted_D is None: # stacking process has not yet begun
            sorted_D = D[cluster_labels==cluster]
        else: # normal stack
            sorted_D = np.vstack((sorted_D, D[cluster_labels==cluster]))
    plt.figure(figsize=(5,5), dpi=100)
    plt.imshow(sorted_D, cmap='gray')
    plt.title(f'Clustered Dist. Matrix for {cluster_name} w/ D{distance_type}')
    plt.colorbar()
    plt.show();