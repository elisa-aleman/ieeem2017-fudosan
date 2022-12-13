#-*- coding: utf-8 -*-
#!python3

#Run in server

import numpy
# Apparently pyclustering.cluster.xmeans.xmeans uses matplotlib
# Which fails to display on server display with GDK so, set to Pdf
import matplotlib 
matplotlib.use('Pdf')
from pyclustering.cluster.xmeans import xmeans #, splitting_type #used in _init_ for xmeans, but is included default anyway
from random import random
import csv

#Print to terminal and to file at the same time
def printSTDlog(strlog, log_file): 
    with open(log_file, 'a') as logf:
            print(strlog)
            strlog+= "\n"
            logf.write(strlog)

# Read a CSV file
def readCSV(filename, titlesplit=True):
    f = open(filename, 'rt', encoding='utf-8')
    reader = csv.reader(f, delimiter = ',')
    table = []
    for row in reader:
        table.append(tuple(row))
    f.close()
    if titlesplit:
        titles = table[0]
        table = table[1:]
        return titles, table
    else:
        return table

def cluster_xmeans(data, kmax=20, tolerance=0.025):
    ## data must be list of lists or list of tuples
    if type(data)==type(numpy.array([])):
        data = data.tolist()
    # Define the classifier
    xmeans_clf = xmeans(
        data, #list of lists or list of tuples
        initial_centers = None, #Default is None. If None, executes random initialization
        kmax=kmax, #Maximum number of clusters
        tolerance=tolerance, #If change of center of cluster is less than tolerance it stops the algorithm
        # criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION, #in-libary variable, default is to use bayesian method, so technically don't need to define when instancing class
        ccore = True #If true, C++ library is used instead of pure python
        )
    # Actually run the clustering process
    xmeans_clf.process()
    # Collect list of clusters
    clusters = xmeans_clf.get_clusters() #each cluster is a list of indexes pointing to the original data.
    return clusters

def cluster_to_ids(clusters, ids):
    # Creating dict for ids vs cluster name
    cluster_labels = []
    for cnum, cluster in enumerate(clusters):
        current_labels = [(ids[ind],cnum) for ind in cluster]
        cluster_labels += current_labels
    cluster_labels = dict(cluster_labels)
    cluster_list = [(cur_id,cluster_labels.get(cur_id)) for cur_id in ids]
    return cluster_list

def main(kmax = 3):
    data_path = '../data/23_words.csv'
    data = readCSV(data_path, titlesplit=False)
    ids = [row[0] for row in data]
    data = [[float(val) for val in row[1:]] for row in data]
    raw_clusters = cluster_xmeans(data, kmax=kmax)
    clusters = cluster_to_ids(raw_clusters, ids)
    log_file = '../logs/clusters_kmax3.csv'
    printSTDlog('City,clusterID', log_file)
    for ins_row in clusters:
        strlog = ','.join([str(val) for val in ins_row])
        printSTDlog(strlog,log_file)


if __name__ == '__main__':
    # main(kmax = 23)
    main(kmax = 3)