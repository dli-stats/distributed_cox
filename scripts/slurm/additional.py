import collections
import itertools

Param = collections.namedtuple("Param", "N K nk p")

# Plan 1
plan1_settings = [
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
]

# Plan 2
plan2_settings = [
    Param(60, 3, (20,) * 3, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(150, 3, (50,) * 3, "Ber(0.5); N(0,1); Ber(0.5)"),
    Param(300, 3, (100,) * 3, "Ber(0.5); N(0,1); Ber(0.5)"),
]


# Plan 3
def plan3_setting(args):
  nk, K, x_dim = args
  nk = nk * x_dim
  N = nk * K
  p = "Ber(0.5)" if x_dim == 1 else "Ber(0.5); N(0, 1)"
  return Param(N=N, K=K, nk=nk, p=p)


plan3_settings = list(
    map(plan3_setting, itertools.product(range(30, 60, 10), [3, 5], [1, 2])))

_settings = plan1_settings + plan2_settings + plan3_settings

settings = []
batch_size = 32
T_star_factorss = ["None"]
eqs = ["eq1", "eq2", "eq3", "eq4", "meta_analysis", "meta_analysis_univariate"]
for (eq, (N, K, nk, p),
     T_star_factors) in itertools.product(eqs, _settings, T_star_factorss):
  if eq == "meta_analysis_univariate":
    eq = "meta_analysis"
    meta_analysis_univariate = True
  else:
    meta_analysis_univariate = False
  x_dim = p.count(";") + 1
  p = p.replace(";", ",").replace(" ", "")
  p = p.replace("Ber", "bernoulli").replace("N", "normal")
  setting = dict(
      eq=eq,
      batch_size=batch_size,
      data=dict(
          N=N,
          K=K,
          X_DIM=x_dim,
          T_star_factors=T_star_factors,
          group_labels_generator_kind=f'custom{nk}',
          group_X=f'custom([[{p}]],None,None)',
      ),
      distributed=dict(
          hessian_use_taylor=True,
          taylor_order=1,
      ),
      meta_analysis=dict(univariate=meta_analysis_univariate),
  )
  settings.append(setting)
