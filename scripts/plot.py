import argparse
import functools
import os.path

from flexutils.io import serialize
from matplotlib import pyplot as plt

from utils.builders.config import build_config
from utils.builders.resume import get_checkpoint_path
from utils.charts import BarChart, Chart2D, CurveChart


def set_title(chart: Chart2D, config):
  if config is None:
    return
  if isinstance(config, str):
    chart.title = config
  elif isinstance(config, dict):
    chart.title = config['name']
    chart.title_kwargs = getattr(config, 'kwargs', dict())
  else:
    raise ValueError(f"Invalid title config {config}")


def set_ylabel(chart: Chart2D, config):
  if config is None:
    return
  if isinstance(config, str):
    chart.y_label = config
  elif isinstance(config, dict):
    chart.y_label = config['name']
    chart.y_label_kwargs = getattr(config, 'kwargs', dict())
  else:
    raise ValueError(f"Invalid ylabel config {config}")


def set_xlabel(chart: Chart2D, config):
  if config is None:
    return
  if isinstance(config, str):
    chart.x_label = config
  elif isinstance(config, dict):
    chart.x_label = config['name']
    chart.x_label_kwargs = getattr(config, 'kwargs', dict())
  else:
    raise ValueError(f"Invalid xlabel config {config}")


def read_last_table_field(config):
  tables_path = os.path.join(get_checkpoint_path(config.path), os.path.pardir,
                             os.path.pardir, "tables")

  values = []
  for table_file in os.listdir(tables_path):
    epoch, extension = os.path.splitext(table_file)
    if not epoch.isdigit() or extension != ".json":
      continue
    epoch = int(epoch)
    table = serialize.load_json(os.path.join(tables_path, table_file))
    values.append((epoch, table[config.field]))

  if not values:
    raise RuntimeError(f"Cannot find any valid table at {tables_path}")

  values = sorted(values)
  return values[-1][1]


def read_all_epochs(config):
  tables_path = os.path.join(get_checkpoint_path(config.path), os.path.pardir,
                             os.path.pardir, "tables")
  values = {}
  for table_file in os.listdir(tables_path):
    epoch, extension = os.path.splitext(table_file)
    if not epoch.isdigit() or extension != ".json":
      continue
    epoch = int(epoch)
    table = serialize.load_json(os.path.join(tables_path, table_file))
    values[epoch] = table[config.field]

  return values


def read_tensor_board_json(config):
  values = {}
  for stamp, global_step, value in serialize.load_json(config.path):
    values[global_step] = value
  return values


def get_data(config):
  if config.type == 'ReadLastTableField':
    return read_last_table_field(config)
  elif config.type == 'ReadAllEpochs':
    return read_all_epochs(config)
  elif config.type == 'ReadTensorBoardJSON':
    return read_tensor_board_json(config)
  else:
    raise ValueError(f"Invalid series type {config.type}")


def plot_bar(axes: plt.Axes, config):
  chart = BarChart()
  set_title(chart, config.title)
  set_ylabel(chart, getattr(config, 'ylabel', None))
  set_xlabel(chart, getattr(config, 'xlabel', None))

  chart.inter_group_margin = config.margin.inter
  chart.intra_group_margin = config.margin.intra
  chart.value_format = "%.3g"
  chart.x_ticks_kwargs = getattr(config, 'xticks', dict())

  legend_config = getattr(config, 'legend', False)
  if legend_config:
    chart.show_legend = True
    if isinstance(legend_config, dict):
      chart.legend_kwargs = legend_config

  bar_labels_config = getattr(config, 'bar_labels', None)
  if bar_labels_config is not None:
    chart.show_bar_labels = True
    chart.bar_label_kwargs['fmt'] = '%.3g'
    if isinstance(bar_labels_config, dict):
      chart.bar_label_kwargs.update(bar_labels_config)

  for index, series_config in enumerate(config.series):
    series = dict()
    if len(config.groups) != len(series_config.data):
      raise ValueError(f"The number of groups ({len(config.groups)}) does not "
                       f"equal to the number of data entries in the series "
                       f"({len(series_config.data)})")
    for group, data_config in zip(config.groups, series_config.data):
      series[group] = get_data(data_config)
    chart.add_series_by_dict(series,
                             legend=getattr(series_config, "legend", None))

  chart.plot(axes)


def plot_curve(axes: plt.Axes, config):
  chart = CurveChart()
  set_title(chart, config.title)
  set_ylabel(chart, getattr(config, 'ylabel', None))
  set_xlabel(chart, getattr(config, 'xlabel', None))
  chart.use_log_y_scale = True

  chart.value_format = "%.3g"
  chart.x_ticks_kwargs = getattr(config, 'xticks', dict())

  chart.show_x_grid = getattr(config, "x_grid", False)
  chart.show_y_grid = getattr(config, "y_grid", False)
  if getattr(config, "grid", False):
    chart.show_x_grid = True
    chart.show_y_grid = True

  legend_config = getattr(config, 'legend', False)
  if legend_config:
    chart.show_legend = True
    if isinstance(legend_config, dict):
      chart.legend_kwargs = legend_config

  for index, series_config in enumerate(config.series):
    data_config = series_config.data
    series = get_data(data_config)
    chart.add_series_by_dict(series,
                             legend=getattr(series_config, "legend", None))

  chart.plot(axes)


def main(args):
  config = build_config(args.config_path)
  figure: plt.Figure = plt.figure(figsize=config.figsize, dpi=config.dpi)
  if config.nrows * config.ncols != len(config.figures):
    raise ValueError("nrows * ncols does not equal to the number of figures")

  for index, figure_config in enumerate(config.figures):
    axes: plt.Axes = figure.add_subplot(config.nrows, config.ncols, index + 1)
    if figure_config.type == 'Bar':
      plot_bar(axes, figure_config)
    elif figure_config.type == 'Curve':
      plot_curve(axes, figure_config)
    else:
      raise ValueError(f"Invalid figure type {figure_config.type}")

  figure.set_tight_layout(tight=True)
  if args.output_path is None:
    plt.show()
  else:
    figure.savefig(args.output_path)


def parse_args():
  formatter_class = functools.partial(argparse.HelpFormatter,
                                      max_help_position=32)
  parser = argparse.ArgumentParser(formatter_class=formatter_class)
  parser.add_argument("-C",
                      "--config_path",
                      type=str,
                      metavar="PATH",
                      required=True,
                      help="path to the figure config file")
  parser.add_argument('-O',
                      '--output_path',
                      type=str,
                      metavar='PATH',
                      help="path to the output figure")

  args = parser.parse_args()
  return args


if __name__ == '__main__':
  main(parse_args())
