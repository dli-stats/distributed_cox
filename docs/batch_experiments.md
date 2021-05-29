# Batch Experiments

We use [sacred](https://github.com/IDSIA/sacred) to track all the batch experiments.
The batch experiment interface defines a set of available configurations.
You can check the default configuration by the following command:
```shell
$ python -m distributed_cox.experiments.run print_config
```
You should see something similar to the following:
```
$ python -m distributed_cox.experiments.run print_config
Configuration (modified, added, typechanged, doc):
  batch_size = 256
  data_generation_key = array([2718843009, 1272950319], dtype=uint32)
  method = 'unstratified_pooled'
  experiment_rand_key = array([4146024105,  967050713], dtype=uint32)
  num_experiments = 10000
  num_threads = 1
  return_result = False
  save_data_csv = None
  save_interval = 50
  seed = 0                           # the random seed for this experiment
  data:
    K = 3
    N = 500
    T_star_factors = None
    X_DIM = 3
    exp_scale = 3.5
    group_X = 'same'
    group_labels_generator_kind = 'same'
    npz_path = None
  distributed:                       # groupped_configs
    hessian_use_taylor = True
    pt2_use_average_guess = False
    taylor_order = 1
  meta_analysis:                     # meta_analysis
    univariate = False
    use_only_dims = None
  solver:
    loglik_eps = 1e-05
    max_num_steps = 40
    score_norm_eps = 0.001
```
The configuration above says that it will run `method="unstratified_pooled"` (which is the classic, non-distributed Cox model), by `num_experiments=10000` repeated trials.
Each segment in the configuration controls some aspect of the experiment detail. For example, the `data:` segment controls the data generation model.

Note that since this is an _Experiment_ interface, the configuration interface is likely to change over time.
For an update-to-date interface, please check `distributed_cox/experiments/run.py` for more details.
