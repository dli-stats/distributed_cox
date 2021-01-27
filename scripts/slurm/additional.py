import collections
import subprocess
import itertools

Param = collections.namedtuple("Param", "N K nk p")
settings = [
    ## Plan 1
    Param(500, 5, (100,) * 5, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 5, (50,) * 3 + (175,) * 2, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 10, (50,) * 10, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 10, (25,) * 5 + (75,) * 5, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(500, 3, (166, 167, 167),
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    Param(500, 3, (83, 166, 255),
          "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7)"),
    Param(
        500, 3, (166, 167, 167),
        "Ber(0.5); N(0,1); Ber(0.3); N(0,0.25); Ber(0.7); N(0, 0.5); Ber(0.2)"),
    Param(
        500, 3, (83, 166, 255),
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
  p = p.replace(";", ",").replace("Ber", "bernoulli").replace("N", "normal")
  cmd = [
      "sbatch",
      "single_longer.sh",
      eq,
      "with",
      "base.num_experiments=10000",
      f"base.X_DIM={x_dim}",
      f"base.batch_size={batch_size}",
      f"data.N={N}",
      f"data.K={K}",
      f'data.group_labels_generator_kind="custom{nk}"',
      f'data.group_X="custom([[{p}]],[],[])"',
      f'data.T_star_factors="{T_star_factors}"',
      f"meta_analysis.univariate={meta_analysis_univariate}",
  ]
  subprocess.check_call(cmd)
