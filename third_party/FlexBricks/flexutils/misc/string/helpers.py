__all__ = [
    'remove_suffix', 'remove_prefix', 'format_table', 'number_ordinal',
    'filtered_join'
]

from typing import Collection, Optional, Callable


def remove_suffix(string: str, suffix: str):
  if string.endswith(suffix):
    return string[:len(string) - len(suffix)]
  return string


def remove_prefix(string: str, prefix: str):
  if string.startswith(prefix):
    return string[len(prefix):]
  return string


def filtered_join(delimiter: str,
                  strings: Collection[Optional[str]],
                  filter_fn: Optional[Callable[[Optional[str]], bool]] = None):
  if filter_fn is None:
    filter_fn = bool
  return delimiter.join([string for string in strings if filter_fn(string)])


def add_optional_suffix(string: str, delimiter: str, suffix: Optional[str]):
  if not suffix:
    return string + delimiter + suffix
  return string


def format_table(headers: Collection[str], rows: Collection[Collection[str]]):
  column_widths = [0] * len(headers)

  rows = list(rows)
  rows.insert(0, headers)
  for i in range(len(rows)):
    rows[i] = list(rows[i])

  num_rows = len(rows)
  num_cols = len(headers)

  for j in range(num_cols):
    for i in range(num_rows):
      column_widths[j] = max(column_widths[j], len(rows[i][j]))

  formatted_rows = []
  for i, row in enumerate(rows):
    formatted_row = "|"
    for j, cell in enumerate(row):
      formatted_row += f' {cell:{column_widths[j]}s} |'
    formatted_rows.append(formatted_row)

    if i == 0:
      formatted_row = '+'
      for j in range(num_cols):
        formatted_row += '=' + ('=' * column_widths[j]) + '=+'
      formatted_rows.append(formatted_row)

  return '\n'.join(formatted_rows)


def number_ordinal(n: int):
  """
  Convert an integer into its ordinal representation::

  >>> number_ordinal(0)   # => '0th'
  >>> number_ordinal(3)   # => '3rd'
  >>> number_ordinal(122) # => '122nd'
  >>> number_ordinal(213) # => '213th'
  """
  if 11 <= (n % 100) <= 13:
    suffix = 'th'
  else:
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
  return str(n) + suffix
