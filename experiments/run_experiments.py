#!/usr/bin/env python

from __future__ import print_function

import os
import subprocess
import tempfile

from generate_big_joins import create_join_workload

script_dir            = os.path.dirname(os.path.realpath(__file__))
output_dir            = os.path.join(script_dir, "output")
executable_default    = "java -Xmx3200m -jar ../simpledb/dist/simpledb.jar".split(" ")
executable_longimpute = "java -Xmx3200m -Dsimpledb.ImputeSlow -jar ../simpledb/dist/simpledb.jar".split(" ")
catalog_default       = "../simpledb/catalog.txt"
queries_default       = "./queries.txt"

def run_experiment_aggregates():

    iters     = 220
    min_alpha = 0.00
    max_alpha = 1.00
    step      = 0.499999

    queries = queries_default

    this_output_dir = os.path.join(output_dir, "regression_tree")
    run_experiment(this_output_dir, iters, min_alpha, max_alpha, step,
                   queries=queries, executable=executable_longimpute,
                   imputationMethod="REGRESSION_TREE")

    this_output_dir = os.path.join(output_dir, "mean")
    run_experiment(this_output_dir, iters, min_alpha, max_alpha, step,
                   queries=queries, executable=executable_longimpute,
                   imputationMethod="MEAN")

    this_output_dir = os.path.join(output_dir, "hot_deck")
    run_experiment(this_output_dir, iters, min_alpha, max_alpha, step,
                   queries=queries, executable=executable_longimpute,
                   imputationMethod="HOTDECK")

def run_experiment_count():

    iters     = 220
    min_alpha = 0.00
    max_alpha = 1.00
    step      = 0.499999

    queries = "queries_count.txt"

    this_output_dir = os.path.join(output_dir, "regression_tree")
    run_experiment(this_output_dir, iters, min_alpha, max_alpha, step,
                   queries=queries, executable=executable_longimpute,
                   imputationMethod="REGRESSION_TREE")

    this_output_dir = os.path.join(output_dir, "mean")
    run_experiment(this_output_dir, iters, min_alpha, max_alpha, step,
                   queries=queries, executable=executable_longimpute,
                   imputationMethod="MEAN")

    this_output_dir = os.path.join(output_dir, "hot_deck")
    run_experiment(this_output_dir, iters, min_alpha, max_alpha, step,
                   queries=queries, executable=executable_longimpute,
                   imputationMethod="HOTDECK")

def run_experiment_acs():
    catalog = catalog_default
    executable = executable_longimpute

    this_output_dir = os.path.join(output_dir, "acs")


    # Write acs query to temporary file
    (f, acs_query) = tempfile.mkstemp()
    os.write(f, "SELECT AVG(c0) FROM acs_dirty;\n")

    # Impute on base table
    iters = 1

    print("Running acs base...")
    cmd = (executable + ["experiment", catalog, acs_query, this_output_dir,
                         str(iters), "--base"])
    print(cmd)
    subprocess.call(cmd)
    print("Running acs base...done.")

    # Impute using ImputeDB
    iters     = 220
    min_alpha = 0.00
    max_alpha = 1.00
    step      = 1.00

    cmd = (executable + ["experiment", catalog, acs_query, this_output_dir,
                         str(iters), str(min_alpha), str(max_alpha),
                         str(step)])
    subprocess.call(cmd)

    os.close(f)

def run_experiment_joins():
    join_output_dir = os.path.join(output_dir, "joins")

    iters     = 20
    min_alpha = 0.00
    max_alpha = 1.00
    step      = 1.00
    # parameters specific to join workload
    # number of queries to generate and evaluate per size of join
    nqueries   = 5
    # minimum number of joins
    min_njoins = 2
    # maximum number of joins
    max_njoins = 8

    # evaluate each size of join separately
    for njoins in range(min_njoins, max_njoins + 1):
      print("Running join experiments. N-joins %d" % njoins)
      # create sub directory for each size of joins
      this_output_dir = os.path.join(join_output_dir, str(njoins))
      # create workload, written out to base directory
      workload = create_join_workload(njoins, nqueries)
      # local file with queries
      queries = "joins_n%d_queries.txt" % njoins
      with open(queries, 'w') as f:
        f.write(workload)

      # execute this size of n joins
      run_experiment(this_output_dir, iters, min_alpha, max_alpha, step, queries = queries, plan_only = True)


def run_experiment(this_output_dir, iters, min_alpha, max_alpha, step,
                   queries=None, executable=None, plan_only=False,
                   imputationMethod=""):
    if not os.path.isdir(this_output_dir):
        os.makedirs(this_output_dir)

    if not queries:
        queries = queries_default

    if not executable:
        executable = executable_default


    catalog = catalog_default

    if imputationMethod:
        imputationMethodOpt = ["--imputationMethod={}".format(imputationMethod)]
    else:
        imputationMethodOpt = []

    planOnlyOpt = ["--planOnly={}".format(plan_only)]

    # Timing using ImputeDB
    subprocess.call(executable +
        ["experiment", catalog, queries, this_output_dir,
         str(iters), str(min_alpha), str(max_alpha), str(step)] + planOnlyOpt +
        imputationMethodOpt)

    # Timing using impute on base table
    if not plan_only:
      subprocess.call(executable + ["experiment", catalog, queries,
          this_output_dir, str(iters), "--base"] +
          imputationMethodOpt)

if __name__ == "__main__":
    import fire
    fire.Fire({
        "experiment-aggregates" : run_experiment_aggregates,
        "experiment-count" : run_experiment_count,
        "experiment-acs" : run_experiment_acs,
        "experiment-joins" : run_experiment_joins,
    })
