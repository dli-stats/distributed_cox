import numpy as np
import torch
from torch.utils import tensorboard
import logging
logger = logging.getLogger("varderiv")
logger.setLevel(logging.DEBUG)

floatt=np.float32
N = 10000
X_DIM = 2
K = 10


def default_X_generator(dim, N):
  if dim % 3 == 0:
    return np.random.binomial(1, 0.5, size=N)
  elif dim % 3 == 1:
    return np.random.uniform(0, 1, size=N)
  elif dim % 3 == 2:
    return np.random.normal(size=N)


def generate_data(N, X_dim, X_generator=default_X_generator, T_thres_ratio=0.6):
  r"""Generates dummy data.

  The generative process works as follow:
    1. Sample beta
    2. Sample X with X_generator
    3. Sample T^* for each X
    4. Sample C according to `\frac{\log(u)}{\exp(X \cdot \beta)}` where `u \sim \text{Unif}(0, 1)`
    5. Reorder X by `T = \min(T^*, C)`
  """
  beta = np.arange(1, X_dim + 1, dtype=floatt)
  X = np.zeros((N, X_dim), dtype=floatt)
  for dim in range(X_dim):
    X[:, dim] = X_generator(dim, N)

  u = np.random.uniform(0, 1, size=N)
  T_star = - np.log(u) / np.exp(X.dot(beta))

  C = np.random.exponential(scale=3.5, size=N)
  delta = T_star <= C

  T = np.minimum(T_star, C)

  sorted_idx = np.argsort(-T) # sort T descending

  T = np.take(T, sorted_idx, axis=0)
  X = np.take(X, sorted_idx, axis=0)
  delta = np.take(delta, sorted_idx, axis=0)

  return X, delta, beta


def run_optimize_step_loop(params,
                           output_fn, 
                           loss_fn, 
                           loss_stop_thres, 
                           log_every_n_steps, 
                           decay_lr_every_n_steps,
                           writer):
  learning_rate = 1e-3
  optimizer = torch.optim.Adam(params, lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

  # Gradient descent loop
  step = 0
  loss = np.inf
  while loss > loss_stop_thres:
    optimizer.zero_grad()
    output = output_fn()
    loss = loss_fn(output)
    
    loss.backward()
    optimizer.step()

    if step % log_every_n_steps == 0:
      if writer is not None:
        writer.add_scalar('loss', loss, step)
      logger.info("Step {}: Loss={}".format(step, loss))

    if step % decay_lr_every_n_steps == 0:
      scheduler.step(loss)

    step += 1

  return params


def _compute_frac_term(X, J, beta):
  e_beta_X_l = torch.exp(torch.mv(X, beta)).view(-1, 1)
  X_dot_e_beta_X_l = X * e_beta_X_l

  e_beta_X_l_cs = torch.cumsum(e_beta_X_l, 0)
  X_dot_e_beta_X_l_cs = torch.cumsum(X_dot_e_beta_X_l, 0)

  return X_dot_e_beta_X_l_cs.index_select(0, J), e_beta_X_l_cs.index_select(0, J)


def eq1(X, J, beta):
  num, denom = _compute_frac_term(X, J, beta)
  frac_term = num / denom

  sum_inner = X.index_select(0, J) - frac_term

  ret = torch.sum(sum_inner, dim=0)

  return ret


def optim_eq1(X, delta, beta_correct, 
              loss_stop_thres=1e-6,
              log_every_n_steps=100,
              decay_lr_every_n_steps=100,
              writer=None):
  """Optimizes Eq 1.
  
  Args:
    - X: Data samples.
    - delta: Mask X over highlighted samples.
    - beta_correct: beta used to generate X.
    - loss_stop_thres: thresold of loss for when to stop optimize.

  Returns:
    Optimal beta.
  """

  N = X.shape[0]

  def loss_fn(output):
    return torch.sum(torch.abs(output), dim=0) / N

  X = torch.from_numpy(X).to(device)
  J = torch.from_numpy(np.where(delta)[0]).long().to(device)
  beta_correct = torch.from_numpy(beta_correct).to(device)

  del delta

  beta = torch.randn(X.shape[1], dtype=X.dtype, requires_grad=True)

  output_correct = eq1(X, J, beta_correct)
  loss_correct = loss_fn(output_correct)
  logger.info("Beta correct eq1 = {} loss = {}".format(output_correct, loss_correct))

  beta = run_optimize_step_loop(
    [beta],
    lambda: eq1(X, J, beta),
    loss_fn,
    loss_stop_thres, 
    log_every_n_steps, 
    decay_lr_every_n_steps,
    writer)[0]

  return beta


def _precompute_eq2_terms(X, J, group_labels, beta_hat):

  beta_hat_grouped = torch.index_select(beta_hat, 0, group_labels)
  
  beta_X = torch.einsum("bi,bi->b", X, beta_hat_grouped)

  e_beta_X = torch.exp(beta_X).view(-1, 1)

  X_e_beta_X = X * e_beta_X

  XX_e_beta_X = torch.einsum('bi,bj,bk->bij', (X, X, e_beta_X))

  e_beta_X_cs_is = torch.cumsum(e_beta_X, 0).index_select(0, J)
  X_e_beta_X_cs_is = torch.cumsum(X_e_beta_X, 0).index_select(0, J)

  return e_beta_X, X_e_beta_X, XX_e_beta_X, e_beta_X_cs_is, X_e_beta_X_cs_is, beta_hat_grouped


def eq2(X, J, beta, group_labels, pre_computed):
  (e_beta_X, X_e_beta_X, 
    XX_e_beta_X, e_beta_X_cs_is, 
    X_e_beta_X_cs_is, beta_hat_grouped) = pre_computed

  beta_sub_beta_hat = beta - beta_hat_grouped
  xxebxbmb = torch.einsum("bij,bj->bi", (XX_e_beta_X, beta_sub_beta_hat))
  xebxbmb = torch.einsum("bi,bi->b", (X_e_beta_X, beta_sub_beta_hat))

  xxebxbmb_cs = torch.cumsum(xxebxbmb, 0)
  xebxbmb_cs = torch.cumsum(xebxbmb, 0).view(-1, 1)

  frac_term = torch.sum(
    (X_e_beta_X_cs_is + xxebxbmb_cs.index_select(0, J)) / 
    (e_beta_X_cs_is + xebxbmb_cs.index_select(0, J)),
    dim=0
  )

  XJ = torch.sum(X.index_select(0, J), dim=0)
  
  return XJ - frac_term



def optim_eq2(X, delta, beta_correct,
              group_labels=None,
              K=K,
              loss_stop_thres=1e-5,
              log_every_n_steps=100,
              decay_lr_every_n_steps=100,
              writer=None):
  N = X.shape[0]

  if group_labels is None:
    assert K is not None
    group_labels = np.random.choice(K, size=N, replace=True, p=[1./K]*K)
  
  K = np.max(group_labels) + 1

  beta_hat = [None] * K

  for k in range(K):
    group_mask = group_labels == k
    X_group = X[group_mask]
    delta_group = delta[group_mask]
    logger.info("Solving Eq 1. for group {} of size {}".format(k, len(X_group)))
    # writer = tensorboard.SummaryWriter(comment="eq2|N={}|X_dim={}|k={}".format(N, X_DIM, k))
    beta_hat[k] = optim_eq1(
      X_group, delta_group, beta_correct,
      loss_stop_thres=loss_stop_thres, 
      log_every_n_steps=log_every_n_steps,
      decay_lr_every_n_steps=decay_lr_every_n_steps, 
      writer=None).detach()
    # writer.close()
    print("Beta for group {} is {}".format(k, beta_hat[k]))

  beta_hat = torch.stack(beta_hat)

  group_labels = torch.from_numpy(group_labels).long().to(device)
  X = torch.from_numpy(X).to(device)
  J = torch.from_numpy(np.where(delta)[0]).long().to(device)
  del delta

  def loss_fn(output):
    return torch.sum(torch.abs(output), dim=0) / N

  beta = beta_hat.mean(dim=0).clone().detach().requires_grad_(True)
  # beta = torch.randn(X.shape[1], dtype=X.dtype, requires_grad=True)

  pre_computed = _precompute_eq2_terms(X, J, group_labels, beta_hat)
  
  try:
    beta = run_optimize_step_loop(
      [beta],
      lambda: eq2(X, J, beta, group_labels, pre_computed),
      loss_fn,
      loss_stop_thres,
      log_every_n_steps, 
      decay_lr_every_n_steps,
      writer)[0]
  except KeyboardInterrupt:
    print("Interrupted, returning current beta")
    pass

  return beta


if __name__ == "__main__":
  disable_cuda = False
  if not disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  X, delta, beta_correct = generate_data(N, X_DIM)

  beta = optim_eq2(X, delta, beta_correct)

# beta = optim_eq1(X, delta, beta_correct)

# writer = tensorboard.SummaryWriter(comment="N={}|X_dim={}".format(N, X_DIM))
# writer.close()
