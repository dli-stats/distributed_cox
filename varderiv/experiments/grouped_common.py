"""Common configs for groupped experiments."""

from sacred import Ingredient

ingredient = Ingredient('grouped')

# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring


@ingredient.config
def config():
  K = 3
  group_labels_generator_kind = "same"
  group_labels_generator_kind_kwargs = {}


@ingredient.named_config
def group_labels_same():
  group_labels_generator_kind = "same"
  group_labels_generator_kind_kwargs = {}


@ingredient.named_config
def group_labels_random():
  group_labels_generator_kind = "random"
  group_labels_generator_kind_kwargs = {}


@ingredient.named_config
def group_labels_arithmetic_sequence():
  group_labels_generator_kind = "arithemetic_sequence"
  group_labels_generator_kind_kwargs = dict(start_val=50)


@ingredient.named_config
def group_labels_single_ladder():
  group_labels_generator_kind = "single_ladder"
  group_labels_generator_kind_kwargs = dict(
      start_val=50,
      repeat_start=1,
  )
