import itertools

group_X_setting1 = "same"
group_X_setting3 = "custom([[normal(0, 1.), normal(0, 0.04), bernoulli(0.5)], [bernoulli(0.1), normal(0, 0.5), normal(2, 0.5)], [bernoulli(0.9), normal(0, 0.04), normal(-1, 1.5)]], [[0, 1]], [3., 5., 1./3])"

eqs = ["eq1", "eq2", "eq3", "eq4", "meta_analysis", "meta_analysis_univariate"]

N_and_group_X_settings_and_taylor_order = [(N, group_X_setting1, taylor_order)
                                           for taylor_order in range(1, 5)
                                           for N in [500, 1000, 3000]
                                          ] + [(3000, group_X_setting3, 1)]

group_labels_generator_kinds = ["same", "arithmetic_sequence"]
T_star_factorss = ["None", "fixed", "gamma(1, 1.)"]
batch_size = 16

settings = []
for (
    (N, group_X_setting, taylor_order),
    eq,
    T_star_factors,
    group_labels_generator_kind,
) in itertools.product(N_and_group_X_settings_and_taylor_order, eqs,
                       T_star_factorss, group_labels_generator_kinds):
  if eq == "meta_analysis_univariate":
    eq = "meta_analysis"
    meta_analysis_univariate = True
  else:
    meta_analysis_univariate = False
  setting = dict(
      eq=eq,
      batch_size=batch_size,
      data=dict(
          N=N,
          K=3,
          X_DIM=3,
          T_star_factors=T_star_factors,
          group_labels_generator_kind=group_labels_generator_kind,
          group_X=group_X_setting,
      ),
      distributed=dict(
          hessian_use_taylor=True,
          taylor_order=taylor_order,
      ),
      meta_analysis=dict(univariate=meta_analysis_univariate),
  )
  settings.append(setting)
