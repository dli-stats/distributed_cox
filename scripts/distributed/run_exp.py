import itertools
import subprocess

results_dir = "realdata_results/"

npz_paths = [f"simulated_data/npzs/dat_{i}/all.npz" for i in (1, 2)]

eqs = [f"eq{i+1}" for i in range(4)
      ] + ["meta_analysis", "meta_analysis_univariate"]

for npz_path, eq in itertools.product(npz_paths, eqs):
  univariate = False
  if eq == "meta_analysis_univariate":
    eq = "meta_analysis"
    univariate = True

  subprocess.check_call([
      "python",
      "-m",
      "distributed_cox.experiments.run",
      "-F",
      results_dir,
      "-p",
      "with",
      f"eq={eq}",
      "num_experiments=1",
      "batch_size=1",
      "solver.max_num_steps=1000",
      f"data.npz_path={npz_path}",
      f"meta_analysis.univariate={univariate}",
  ])
