"""Run additional experiments."""

import collections
import subprocess
import itertools
import textwrap

import simple_slurm

slurm = simple_slurm.Slurm(cpus_per_task=8,
                           nodes=1,
                           time='0-2:00',
                           partition='short',
                           mem=3000,
                           output='hostname_%j.out',
                           error='hostname_%j.err',
                           mail_type='FAIL',
                           mail_user='dl263@hms.harvard.edu')

Param = collections.namedtuple("Param", "N K nk p")
settings = [
    ## Plan 1
    Param(500, 5, (100,) * 5, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 5, (50,) * 3 + (175,) * 2, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 10, (50,) * 10, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 10, (25,) * 5 + (75,) * 5, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 3, (166, 167, 167),
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    Param(500, 3, (83, 166, 251),
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    Param(
        500, 3, (166, 167, 167),
        "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7); N(0, 0.5); Ber(0.2)"),
    Param(
        500, 3, (83, 166, 251),
        "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7); N(0, 0.5); Ber(0.2)"),
    Param(500, 5, (100,) * 5,
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    Param(500, 5, (50,) * 3 + (175,) * 2,
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    Param(500, 10, (50,) * 10,
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    Param(500, 10, (25,) * 5 + (75,) * 5,
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    #
    Param(
        500, 5, (100,) * 5,
        "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7); N(0, 0.5); Ber(0.2)"),
    Param(
        500, 5, (50,) * 3 + (175,) * 2,
        "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7); N(0, 0.5); Ber(0.2)"),
    Param(
        500, 10, (50,) * 10,
        "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7); N(0, 0.5); Ber(0.2)"),
    Param(
        500, 10, (25,) * 5 + (75,) * 5,
        "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7); N(0, 0.5); Ber(0.2)"),
    ## Plan 2
    Param(60, 3, (20,) * 3, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(150, 3, (50,) * 3, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(300, 3, (100,) * 3, "Ber(0.5); N(0,1); Ber(0.5)"),
]

root_dir = subprocess.check_output(["git", "rev-parse",
                                    "--show-toplevel"]).strip().decode("utf-8")
storage_dir = "/n/scratch3/users/d/dl263/varderiv_experiments"

batch_size = 32
T_star_factorss = ["None"]
eqs = ["eq1", "eq2", "eq3", "eq4", "meta_analysis", "meta_analysis_univariate"]
for (eq, (N, K, nk, p),
     T_star_factors) in itertools.product(eqs, settings, T_star_factorss):
  if eq == "meta_analysis_univariate":
    eq = "meta_analysis"
    meta_analysis_univariate = True
  else:
    meta_analysis_univariate = False
  x_dim = p.count(";") + 1
  p = p.replace(";", ",").replace(" ", "")
  p = p.replace("Ber", "bernoulli").replace("N", "normal")

  slurm.sbatch(
      textwrap.dedent(f"""
        ROOT_DIR={root_dir}
        source activate varderiv
        cd $ROOT_DIR

        python -m distributed_cox.experiments.run -p -F {storage_dir} with \\
          num_experiments=10000 \\
          eq={eq} \\
          batch_size={batch_size} \\
          data.X_DIM={x_dim} \\
          data.N={N} \\
          data.K={K} \\
          data.group_labels_generator_kind='custom{nk}' \\
          data.group_X='custom([[{p}]],None,None)' \\
          data.T_star_factors='{T_star_factors}' \\
          meta_analysis.univariate={meta_analysis_univariate}
        """),
      shell="/bin/bash",
  )
