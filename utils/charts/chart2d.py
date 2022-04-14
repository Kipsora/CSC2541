__all__ = ['Chart2D']

from typing import Optional

from matplotlib.axes import Axes


class Chart2D(object):

  def __init__(self):
    self.title: Optional[str] = None
    self.x_label: Optional[str] = None
    self.y_label: Optional[str] = None

    self.use_log_y_scale: bool = False
    self.use_log_x_scale: bool = False

    self.title_kwargs: dict = dict()

    self.x_label_kwargs: dict = dict()
    self.x_ticks_kwargs: dict = dict()

    self.y_label_kwargs: dict = dict()
    self.y_ticks_kwargs: dict = dict()

    self.show_x_grid: bool = False
    self.show_y_grid: bool = False

  def plot(self, axes: Axes):
    if self.title is not None:
      axes.set_title(self.title, **self.title_kwargs)
    if self.x_label is not None:
      axes.set_xlabel(self.x_label, **self.x_label_kwargs)
    if self.y_label is not None:
      axes.set_ylabel(self.y_label, **self.y_label_kwargs)
    if self.show_x_grid:
      axes.xaxis.grid(zorder=-1, linestyle='dashed')
      axes.set_axisbelow(True)
    if self.show_y_grid:
      axes.yaxis.grid(zorder=-1, linestyle='dashed')
      axes.set_axisbelow(True)
    if self.use_log_y_scale:
      axes.set_yscale('log')
    if self.use_log_x_scale:
      axes.set_xscale('log')
