#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: main.py
Author: E. Alcobaça
Email: e.alcobaca@gmail.com
Github: https://github.com/ealcobaca
Description: TODO
"""

import multiprocessing as mp
import sys
import os
import getopt
from scipy.io import arff
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np
from arff2pandas import a2p


# def read_arff_df(path):
#     """TODO: Docstring for read_arff_df.
#
#     :path: TODO
#     :returns: TODO
#
#     """
#     with open(path) as file:
#         data = arff.load(file)
#         attrs = data['attributes']
#         attrs_final = [attr[0] for attr in attrs]
#     return pd.DataFrame(data=data['data'], columns=attrs_final)
#
#


def job(dataset, output):
    """TODO: Docstring for job.

    :id: TODO
    :dataset: TODO
    :returns: TODO

    """
    data_name = dataset[0].split('/')[1]
    report = {"dataset": [],
              "tot-col": [],
              "col-drop-var": [],
              "col-drop-corr": [],
              "col-lda": [],
              "col-pca80": [],
              "col-pca85": [],
              "col-pca90": [],
              "col-pca95": []}
    report = pd.DataFrame(data=report)
    # reading a row dataset
    data_raw = pd.read_csv(dataset[0])

    data_raw_id = data_raw[["id"]]  # get the id
    data_raw_target = data_raw[["Class"]]  # get the id
    data_raw_target_num = np.array(
        pd.get_dummies(data_raw_target).values.argmax(1))  # string to numeric
    data_raw = data_raw.drop(columns=["Class", "id"])  # remove id and target

    data_raw = data_raw.drop(
        columns=data_raw.columns[data_raw.var().abs() == 0])
    data_raw = (data_raw - data_raw.mean())/data_raw.std()  # z-score

    # removing correlated features
    # Create correlation matrix and get the absolute value
    data_raw_corr_matrix = data_raw.corr().abs()
    # Select upper triangle of correlation matrix
    upper_matrix = data_raw_corr_matrix.where(
        np.triu(np.ones(data_raw_corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.99
    to_drop = [column for column in upper_matrix.columns
               if any(upper_matrix[column] > 0.80)]
    # Removing
    #print(data_raw.shape + " " + to_drop.shape)
    print(data_raw.shape[1])
    print(len(to_drop))
    data_raw = data_raw.drop(columns=to_drop)

    # applying the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(data_raw, data_raw_target_num)
    data_transf_lda = lda.transform(data_raw)
    data_transf_lda = pd.DataFrame(data_transf_lda)
    # unite the id, transformed data and target and write on hd
    data = pd.concat([data_raw_id, data_transf_lda, data_raw_target], axis=1)
    # save file
    path = output + "lda/"
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path+data_name, index=False)

    # apllying PCA
    name = dataset[0].split('/')
    for value in [80, 85, 90, 95]:
        pca = PCA(value/100.0)
        pca.fit(data_raw)
        data_transf_pca = pca.transform(data_raw)
        data_transf_pca = pd.DataFrame(data_transf_pca)
        path = output + "pca/" + str(value) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        data.to_csv(path+data_name, index=False)

    print("DONE! -- " + data_name)


def jobs_mult_proc(datasets, output, nprocesses):
    """TODO: Docstring for works_mult_proc.

    :datasets: TODO
    :output: TODO
    :nprocesses: TODO
    :returns: TODO

    """
    #job(datasets[13], output)
    pool = mp.Pool(processes=nprocesses)
    # launching multiple evaluations asynchronously
    multiple_results = [pool.apply_async(job, (dataset, output,))
                        for dataset in datasets]
    pool.close()
    pool.join()
    return


def read_file_datasets(path):
    """TODO: Docstring for read_file_datasets.

    :path: TODO
    :returns: TODO

    """
    try:
        with open(path, "r") as file:
            lines = [line.split() for line in file.readlines()]
    except IOError as err:
        print("I/O error({0}): {1}".format(err.errno, err.strerror))
        print("Execution failed!")
        sys.exit(1)
    except Exception as exce:  # handle other exceptions
        print("Unexpected error:", sys.exc_info()[0])
        print("Execution failed!")
        sys.exit(1)

    return lines


def main(argv):
    """TODO: Docstring for main.

    :argv: TODO
    :returns: TODO

    """
    nprocesses = 1
    output = ''

    try:
        opts, args = getopt.getopt(
            argv, "ho:n:", ["help=", "output=", "nprocesses="])

    except getopt.GetoptError as err:
        print(str(err))
        print("drpy.py [-o <output path> -n <number of processes>] <file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("usage: drpy.py [-o path | -n integer | ...] <file>")
            print("Options and arguments:")
            print(("-h           : print this help"
                   "message and exit (also --help)"))
            print(("-n integer   : process number that it"
                   " will be created in parallel running (also --nprocesses)"))
            print(("-o path      : the output path (also --path)"))
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-n", "--nprocesses"):
            nprocesses = int(arg)

    if not os.path.exists(output):
        os.makedirs(output)

    datasets = read_file_datasets(args[0])
    jobs_mult_proc(datasets, output, nprocesses)
    #print(datasets)

if __name__ == "__main__":
    main(sys.argv[1:])