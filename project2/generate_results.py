"""
Cluster using my algorithms
"""
import argparse
import os.path
import warnings

import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

purples = ["#0a0612", "#392249", "#482980", "#673ab7",
           "#7a52aa", "#9779bd", "#b59fd0", "#d3c5e3"]

from util import load_data, load_distance_matrices, create_distance_matrices, cluster_size_distribution, SSE, BSS, print_cluster_topics, plot_cluster_size_distributions, visualize_clustering
from sklearn.metrics import silhouette_score, homogeneity_score
from kmedoids import kMedoidsAlgorithm
from completelink import CompleteLinkAlgorithm

clustering_algos = [kMedoidsAlgorithm, CompleteLinkAlgorithm]

def graph_performance_with_k(k, SSE, BSS, sil, distance_measure=1):
    plt.rcParams['figure.dpi'] = 150
    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(9,3)
    # fig.tight_layout()
    #BSS and SSE plotting
    axes[0].plot(k, SSE, color=purples[1], label='SSE(cohe)')
    axes[0].plot(k, BSS, color=purples[5], label='BSS(sep)')
    axes[0].plot(k, (np.array(BSS)+np.array(SSE))/2, '--', color=purples[3], label='Weighted Average')
    axes[0].set_title('k vs. SSE and BSS (Distance {})'.format(str(distance_measure)))
    axes[0].set_ylabel('SSE and BSS')
    axes[0].set_xlabel('k')
    axes[0].legend()

    # silhouette plotting
    axes[1].plot(k, sil, color=purples[4])
    axes[1].set_title('k vs. Silhouette Score (Distance {})'.format(str(distance_measure)))
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_xlabel('k')
    plt.show();


def demonstrate_consistency(cluster_labels_1, cluster_labels_2):
    #entropy
    cluster_labels_1 = np.array(cluster_labels_1)
    cluster_labels_2 = np.array(cluster_labels_2)

    m = np.size(cluster_labels_1)
    k1 = np.max(cluster_labels_1)
    k2 = np.max(cluster_labels_2)
    clusters_1 = np.arange(k1+1, dtype=int)
    clusters_2 = np.arange(k2+1, dtype=int)
    e = 0.0
    purities = []
    for i in clusters_1:
        mi = np.sum((cluster_labels_1==i))
        if mi == 0.0:
            pass
        else:
            p = []
            for j in clusters_2:
                mij = np.sum((cluster_labels_1==i)*(cluster_labels_2==j))
                if mij != 0:
                    p.append(mij/mi)
            ei = entropy(p)
            e += (mi/m)*ei
            purities.append((mi/m)*np.max(p))
    #purity:
    purity = np.sum(purities)
    return e, purity


if __name__ == '__main__':
    """
    Main method.  
    
    Parses args and generates results.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Generate results for our clustering algorithms.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data(should be .txt).')
    parser.add_argument('--regen', dest='regen', action='store_true',
                        help='Opts for the re-generation of the distance matrices as opposed to the use of my pre-generated ones.  Submit flag --regen if you DO want them to be regenerated.')
    parser.add_argument('--tuning', dest='tuning', action='store_true',
                        help='turns on tuning, which will loop through the algos many times to evaluate given hyperparameters.')
    parser.add_argument('--topics', dest='topics', action='store_true',
                        help='turns on topics, which will print out the top ten tokens for the top few clusters.')
    parser.add_argument('--plots', dest='plots', action='store_true',
                        help='Turns on the generation and printing of cluster visualization and cluster size distribution plots.')
    parser.add_argument('--consistency', dest='consistency', action='store_true',
                        help='Turns on the function to use entropy and purity to compare purity.  Should do everything for you.')
    parser.set_defaults(regen=False, tuning=False, topics=False, plots=False, consistency=False)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    regen = args.regen
    tuning = args.tuning
    topics = args.topics
    plots = args.plots
    consistency = args.consistency 
    X,vocab = load_data(data_path) # default: 'data/cnnhealth.txt'

    if regen:
        distance_matrix_1, distance_matrix_2 = create_distance_matrices(X)
    else:
        distance_matrix_1, distance_matrix_2 = load_distance_matrices()

    optimal_ks = (50, 600)

    warnings.filterwarnings("ignore") # WILL SUPRESS WARNING FOR CLEARER OUTPUT
    if not tuning:
        
        for algorithm, k in zip(clustering_algos, optimal_ks):
            algo = algorithm()
            print('================')
            print(f'{algo.name}:')
            print(f'NUM_CLUSTERS: {k}')
            print('================')

            print('For Distance Measure 1:')
            cluster_labels_d1 = algo.cluster(distance_matrix_1, k)
            if plots:
                sizes, frequencies = cluster_size_distribution(cluster_labels_d1)
                print('Distribution will be displayed in plot...')
                plot_cluster_size_distributions(sizes, frequencies, algo.name, '1')
                # print(f'Cluster Sizes: {str(size_distribution)}')
                visualize_clustering(distance_matrix_1, cluster_labels_d1, algo.name, '1')
            else:
                print('submit arg --plots to see distributions + visualizations')
            print(f'Cohesion: SSE:{round(SSE(X,cluster_labels_d1),3)}')
            print(f'Separation: BSS:{round(BSS(X,cluster_labels_d1),3)}')
            print(f"Silhouette Score:{round(silhouette_score(distance_matrix_1, labels=cluster_labels_d1, metric='precomputed'),3)}")
            if topics:
                print_cluster_topics(X, vocab, cluster_labels_d1)
            print(' ')

            print('For Distance Measure 2:')
            cluster_labels_d2 = algo.cluster(distance_matrix_2, k) 
            if plots:
                sizes, frequencies = cluster_size_distribution(cluster_labels_d2)
                print('Distribution will be displayed in plot...')
                plot_cluster_size_distributions(sizes, frequencies, algo.name, '2')
                # print(f'Cluster Sizes: {str(size_distribution)}')
                visualize_clustering(distance_matrix_2, cluster_labels_d2, algo.name, '2')
            else:
                print('submit arg --plots to see distributions + visualizations')
            print(f'Cohesion: SSE:{round(SSE(X,cluster_labels_d2),3)}')
            print(f'Separation: BSS:{round(BSS(X,cluster_labels_d2),3)}')
            print(f"Silhouette Score:{round(silhouette_score(distance_matrix_2, labels=cluster_labels_d2, metric='precomputed'),3)}")
            if topics:
                print_cluster_topics(X, vocab, cluster_labels_d2)
            print(' ')
    else: # this section only runs during parameter tuning:
        for algorithm in clustering_algos:
            algo = algorithm()
            print('================')
            print(f'{algo.name}:')
            print('================')  

            ks = [3,4,5,6,7,8,9]+list(range(10,600,10))
            d1_SSE = []
            d1_BSS = []
            d1_Silhouettes = []

            d2_SSE = []
            d2_BSS = []
            d2_Silhouettes = []

            for k in ks:
                print(f'NUM_CLUSTERS: {k}')
                
                cluster_labels_d1 = algo.cluster(distance_matrix_1, k)
                # size_distribution = cluster_size_distribution(cluster_labels_d1)
                d1_SSE.append(SSE(X,cluster_labels_d1))
                d1_BSS.append(BSS(X,cluster_labels_d1))
                d1_Silhouettes.append(silhouette_score(distance_matrix_1, labels=cluster_labels_d1, metric='precomputed'))

                cluster_labels_d2 = algo.cluster(distance_matrix_2, k)
                # size_distribution = cluster_size_distribution(cluster_labels_d2)
                d2_SSE.append(SSE(X,cluster_labels_d2))
                d2_BSS.append(BSS(X,cluster_labels_d2))
                d2_Silhouettes.append(silhouette_score(distance_matrix_2, labels=cluster_labels_d2, metric='precomputed'))
            
            graph_performance_with_k(ks, d1_SSE, d1_BSS, d1_Silhouettes, distance_measure = 1)
            graph_performance_with_k(ks, d2_SSE, d2_BSS, d2_Silhouettes, distance_measure = 2)
    if consistency:
        algo1 = clustering_algos[0]()
        algo2 = clustering_algos[1]()
        cluster_labels_a1_d2 = algo1.cluster(distance_matrix_2, optimal_ks[0])
        cluster_labels_a2_d2 = algo2.cluster(distance_matrix_2, optimal_ks[1])
        ent1, purity1 = demonstrate_consistency(cluster_labels_a1_d2, cluster_labels_a2_d2)
        ent2, purity2 = demonstrate_consistency(cluster_labels_a2_d2, cluster_labels_a1_d2)
        print(f'Consistency Checking: Comparing {algo1.name} to {algo2.name:}')
        print(f'When {algo1.name} is the label, Entropy = {round(ent2, 3)}, Purity = {round(purity2, 3)}')
        print(f'When {algo2.name} is the label, Entropy = {round(ent1, 3)}, Purity = {round(purity1, 3)}')

        

