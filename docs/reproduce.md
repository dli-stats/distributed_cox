# Reproduce results in paper XXX

To reproduce the results in Table XX, **TODO** using simulated real data [dat_std_simulated.csv](https://github.com/dli-stats/distributed_cox_paper_simudata.git).



# Docker

To simplify the process, we provide a pre-built Docker image to easily reproduce the results.

The pre-built Docker image pre-packages all the softward dependencies, a frozen version of our code for the paper submission, and the generated simulated data.

Using Docker is also highly recommanded if you are using Windows, since JAX, one of our dependencies, does not officially support Windows, and it is complicated to install on Windows.

Please see the steps below to setup the Docker environment.

### Install Docker and pull our pre-built image

If you don't have Docker installed, please install it for your system from [Docker Website]((https://docs.docker.com/get-docker/)).

Then, after starting up Docker (e.g. by clicking on the installed Docker application), run the following command to pull our pre-built Docker image:

```shell
$ docker pull dlistats/distributed_cox_paper_repro
```

### Connect to Docker instance

Once pulling finished, you can connect to a running instance of the image by:
```shell
$ docker run -it dlistats/distributed_cox_paper_repro bash
```
This will connect you to a terminal running on the containerized instance.
You should see a prompt similar to the following:
```bash
$ docker run -it dlistats/distributed_cox_paper_repro bash
root@684d66026829:/distributed_cox#
```
Now type `ls` to list the current directory -- this is root directory for our frozen source code:
```bash
root@684d66026829:/distributed_cox# ls
Dockerfile  README.md	     distributed_cox.code-workspace  notebooks	       scripts	 simulated_data
LICENSE     distributed_cox  distributed_cox.egg-info	     requirements.txt  setup.py
```

Our generated simulated data is located at `/cox-data`, be sure to check the following command:
```
root@684d66026829:/distributed_cox# ls /cox-data/
README.md  dat_std_simulated.csv
```
As you can see, the generated data is the `dat_std_simulated.csv` file.
You can check its contents by
```
root@684d66026829:/distributed_cox# head /cox-data/dat_std_simulated.csv
```

Please refer to [our simulated data format](#simulated-data-format) for more details on this file.

You may now exit the instance anytime by `Ctrl^D`.

Anytime you want to access or run our code, make sure to always first [connect to an instance](#connect-to-docker-instance).

Please note that changes inside the instance will not persist after exiting the instance -- this is desirable for the sake of reproducing the experiments, since it allows us to always start from the fresh frozen snapshot of the code.

You are now ready to reproduce the results in the paper.

## Simulated Data Format

The simulated data is a `.csv` file is deliminated by comma, and contains 20 columns of data.
The `indDP` column marks the group label for each individual sites

**TODO(dli-stats): expand above description**

```
$ head dat_std_simulated.csv
"","time","status","A","X1","X2","X3","X6","X8","X9","X11","X12","X13","X15","X16","X24","X25","X26","indDP"
"1",22,1,1,1,0.138244745134269,1,0,0,1,0,0,1,0,0,-1.32005644874214,0,0.147348806107656,2
"2",30,0,0,1,0.325395161862787,0,0,0,0,0,1,1,0,0,0.407715046771178,0,0.184557548571734,3
"3",30,0,1,0,1.12012703492545,1,1,1,0,0,1,0,0,1,-0.427273637759974,1,1.19999268135355,1
"4",30,0,1,0,0.77154922838838,1,1,1,1,0,1,0,0,1,0.53639594078607,1,0.961339182699883,2
"5",30,0,1,1,-1.78715388641396,0,0,0,0,0,0,0,0,0,-0.403682730054287,1,-1.4685169859299,3
"6",30,0,1,1,0.542950889945675,0,0,1,0,0,1,0,0,0,2.36349715945938,1,0.422280525400782,1
"7",30,0,1,1,0.217590387586867,1,0,0,1,0,1,1,0,0,-0.124941034157083,1,-0.0733960370417061,2
"8",30,0,1,1,0.850009367658679,1,0,1,0,0,1,1,0,0,-1.2764285275839,1,0.820864536928874,2
"9",30,0,1,1,-0.987494509021556,0,0,0,1,0,1,1,0,0,-0.540777786813464,1,-1.10954374625329,1
```

# Custom Installation

For reproducing the experiments, [we recommand using Docker](#docker).

If you like to use our code directly, you may also follow the steps in [installation notes](/docs/install.md) to install the package.

To reproduce the experiments, you also need to download our generated simulated data manually:
```bash
$ export DATA_DIR=/path/to/data_dir/  # A path to store the generated
$ git clone https://github.com/dli-stats/distributed_cox_paper_simudata.git $DATA_DIR
```

Then, change to the repo's root to start reproducing the results
```bash
cd /path/to/distributed_cox
```

# Reproducing the Results

## Data preparation

The first step of reproducing the results is to convert the raw simulated data into a format consumable by our code.

If you are using Docker, this step is not necessary, since we also baked in the converted data at `/distributed_cox/npz_data/` for you.

If you are using a custom install, the following command does the conversion:
```bash
$ export DATA_NPZ_DIR=npz_data/ # A path to store the converted data
$ mkdir -p $DATA_NPZ_DIR
$ python scripts/distributed/convert_simulated_data.py \
  convert_from_csv \
  $DATA_DIR/dat_std_simulated.csv \
  $DATA_NPZ_DIR
```

The command consumes `dat_std_simulated.csv` and outputs several `.npz` files under `$DATA_NPZ_DIR` (by default this is `/distributed_cox/npz_data/` in the Docker instance):
```bash
$ ls $DATA_NPZ_DIR
all.npz  local_0.npz  local_1.npz  local_2.npz
```

The `all.npz` contains all the data combined together.

Each `local_*.npz` contains individual site data.

Note that in a practical setting, one won't have access to `all.npz`, and each `local_*.npz` is scattered at each site.


## Analysis

The analysis code is divided into two kinds: _Batch Experiment_ and _distributed inference_, corresponding to two different entry points.

To briefly explain the difference of the two kinds:

**Batch Experiment** is what we use for generating all the variance analysis in Table X **TODO**.
It's implemented such that it is very efficient to run many trials (10000 trials per experiment in our paper) in batch mode in parallel.

**Distributed Inference** implements the two distributed analysis methods, backed by a simple file-exchange based communication protocal.  Multiple sites and an analysis center can use the _Distributed Inference_ code to perform real world distributed inference. Note that _Batch Experiment_ also implements a version of the distributed analyses in the paper, but these are only tuned for running experiments, and not suitable for multi-party communication (i.e. each experiment is ran on a single computer in a single process, which allowed the implementation to aggressively optimize for performance, for running the trials efficiently).

_Batch Experiment_ implements all the six analysis methods, while _Distributed Inference_ only implements the two distributed analysis methods.
For the sake of reproducing the results for the simulated data, we will use
_Distributed Inference_ for the two distributed analysis methods, and _Batch Experiment_ for the other four methods.

For distributed settings, we will actually run the analyses using the realistic distributed protocal. For other settings, since they are either not distributed, or requires minimal distribution (i.e. the meta-analysis methods, which have only one step of site-center communication), we will simply use the _Batch Experiment_ code (with batch size of $1$).
This makes this experiment more realistic, compared to running all methods using _Batch Experiment_.

Below are short summary of all commands for running all six analysis methods.

### Unstratified Pooled Analysis

```bash
$ python -im distributed_cox.experiments.run \
    with num_experiments=1 batch_size=1 \
    method=unstratified_pooled  \
    return_result=True \
    data.npz_path=$DATA_NPZ_DIR/all.npz
```

### Unstratified Distributed Analysis

```bash
$ python scripts/distributed/run_flow.py run_flow \
  unstratified_distributed $DATA_NPZ_DIR --check_result
```

### Stratified Pooled Analysis

```bash
$ python -im distributed_cox.experiments.run \
    with num_experiments=1 batch_size=1 \
    method=stratified_pooled  \
    return_result=True \
    data.npz_path=$DATA_NPZ_DIR/all.npz
```

### Stratified Distributed Analysis

```bash
$ python scripts/distributed/run_flow.py run_flow \
  stratified_distributed $DATA_NPZ_DIR --check_result
```

### Univariate Meta-Analysis

```bash
$ python -im distributed_cox.experiments.run \
    with num_experiments=1 batch_size=1 \
    method=meta_analysis  \
    meta_analysis.univariate=True \
    return_result=True \
    data.npz_path=$DATA_NPZ_DIR/all.npz
```

### Multivariate Meta-Analysis

```bash
$ python -im distributed_cox.experiments.run \
    with num_experiments=1 batch_size=1 \
    method=meta_analysis  \
    return_result=True \
    data.npz_path=$DATA_NPZ_DIR/all.npz
```