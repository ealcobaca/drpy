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
import math
import time
import sys
import os
import getopt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def preprocessing(data_raw, report):
    """Preprocessing the dataset.
       Docstring for preprocessing.

    :data-raw: dataset with numeric values (dataframe)
    :returns: TODO

    """
    report.append(data_raw.shape[1])  # column length
    data_raw = data_raw.drop(
        columns=data_raw.columns[data_raw.var().abs() == 0])
    report.append(
        data_raw.shape[1])  # column length after drop without variation
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
    data_raw = data_raw.drop(columns=to_drop)
    report.append(
        data_raw.shape[1])  # column length after drop correlated features
    return data_raw


def applyingTSNE(data_raw, n_components, data_raw_id, data_raw_target,
                 data_name, output, report, benchm):
    """TODO: Docstring for applyingTSNE.

    :data_raw: TODO
    :n_components: TODO
    :data_raw_target: TODO
    :data_name: TODO
    :output: TODO
    :returns: TODO

    """
    for n_comp in n_components:
        time_start = time.time()
        porc_comp = int(math.floor(data_raw.shape[1] * (n_comp/100)))
        data_transf_tsne = TSNE(n_components=porc_comp,
                                method="exact").fit_transform(data_raw)
        data_transf_tsne = pd.DataFrame(data_transf_tsne)
        time_end = time.time()
        benchm.append(round((time_end-time_start) * 1000.0, 4))

        report.append(data_transf_tsne.shape[1])
        path = output + "tsne/" + str(n_comp) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        # unite the id, transformed data and target and write on hd
        data = pd.concat(
            [data_raw_id, data_transf_tsne, data_raw_target], axis=1)
        data.to_csv(path+data_name, index=False)

    return


def applyingPCA(data_raw, perc, data_raw_id,
                data_raw_target, data_name, output, report, benchm):
    """this function applies the PCA.

    :data_raw: TODO
    :perc: TODO
    :data_raw_id: TODO
    :data_raw_target: TODO
    :data_name: TODO
    :output: TODO
    :report: TODO
    :returns: TODO
    """

    for value in perc:
        time_start = time.time()
        pca = PCA(value/100.0)
        pca.fit(data_raw)
        data_transf_pca = pca.transform(data_raw)
        data_transf_pca = pd.DataFrame(data_transf_pca)
        time_end = time.time()
        benchm.append(round((time_end-time_start) * 1000.0, 4))

        report.append(data_transf_pca.shape[1])
        path = output + "pca/" + str(value) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        # unite the id, transformed data and target and write on hd
        data = pd.concat(
            [data_raw_id, data_transf_pca, data_raw_target], axis=1)
        data.to_csv(path+data_name, index=False)

    return


def applyingLDA(data_raw, data_raw_target_num, data_raw_id,
                data_raw_target, data_name, output, report, benchm):
    """This function applies LDA in a dataset

    :data_raw: TODO
    :target: TODO
    :output: TODO
    :report: TODO
    :returns: TODO
    """

    time_start = time.time()
    lda = LinearDiscriminantAnalysis()
    lda.fit(data_raw, data_raw_target_num)
    data_transf_lda = lda.transform(data_raw)
    time_end = time.time()
    benchm.append(round((time_end-time_start) * 1000.0, 4))

    data_transf_lda = pd.DataFrame(data_transf_lda)
    report.append(data_transf_lda.shape[1])
    # unite the id, transformed data and target and write on hd
    data = pd.concat([data_raw_id, data_transf_lda, data_raw_target], axis=1)

    # save file
    path = output + "lda/"
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path+data_name, index=False)

    return


def job(dataset, output):
    """jop that will be executed in parallel.

    :dataset: TODO
    :returns: TODO

    """
    data_name = dataset[0].split('/')[1]
    report = []
    benchm = []
    report.append(data_name)
    benchm.append(data_name)
    # reading a row dataset
    data_raw = pd.read_csv(dataset[0])

    data_raw_id = data_raw[["id"]]  # get the id
    data_raw_target = data_raw[["Class"]]  # get the Class
    data_raw_target_num = np.array(
        pd.get_dummies(data_raw_target).values.argmax(1))  # string to numeric
    data_raw = data_raw.drop(columns=["Class", "id"])  # remove id and target

    # preprocessing
    data_raw = preprocessing(data_raw, report)

    # applying the LDA
    applyingLDA(data_raw, data_raw_target_num, data_raw_id,
                data_raw_target, data_name, output, report, benchm)
    # apllying PCA
    perc = [80, 85, 90, 95]
    applyingPCA(data_raw, perc, data_raw_id,
                data_raw_target, data_name, output, report, benchm)
    # applying TSNE
    n_components = [10, 20, 30, 50]
    applyingTSNE(data_raw, n_components, data_raw_id, data_raw_target,
                 data_name, output, report, benchm)
    return report, benchm


def jobs_mult_proc(datasets, output, nprocesses):
    """TODO: Docstring for works_mult_proc.

    :datasets: TODO
    :output: TODO
    :nprocesses: TODO
    :returns: TODO

    """
    report_columns = ["dataset", "tot-col", "col-drop-var", "col-drop-corr",
                      "col-lda", "col-pca80", "col-pca85",
                      "col-pca90", "col-pca95",
                      "col-tsne10", "col-tsne25", "col-tsne50", "col-tsne75"]
    benchm_columns = ["dataset", "time-lda",
                      "time-pca80", "time-pca85", "time-pca90", "time-pca95",
                      "time-tsne10", "time-tsne25",
                      "time-tsne50", "time-tsne75"]

    # job(datasets[13], output)
    pool = mp.Pool(processes=nprocesses)
    # launching multiple evaluations asynchronously
    multiple_results = [pool.apply_async(job, (dataset, output,))
                        for dataset in datasets]
    results = [result.get() for result in multiple_results]
    report = [report[0] for report in results]
    benchm = [time[1] for time in results]

    pool.close()
    pool.join()

    report = pd.DataFrame(data=report, columns=report_columns)
    benchm = pd.DataFrame(data=benchm, columns=benchm_columns)

    path = "report/"
    if not os.path.exists(path):
        os.makedirs(path)
    report.to_csv(path+"report.csv", index=False)
    benchm.to_csv(path+"benchmarking.csv", index=False)

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

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
