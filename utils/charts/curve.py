__all__ = ['CurveChart']

from typing import Optional, Tuple, Dict, Union, List, Sequence

from matplotlib.axes import Axes

from utils.charts import Chart2D, ValueWithVariance


class CurveChart(Chart2D):
  Series = Tuple[Optional[str], Dict, Dict[int, Union[float,
                                                      ValueWithVariance]]]

  def __init__(self):
    super().__init__()

    self.show_legend: bool = False

    self.value_format: str = "%.3g"
    self.legend_kwargs: dict = dict()

    self._baselines: Dict[str, float] = dict()
    self._series: List[CurveChart.Series] = list()

  def add_series_by_dict(self,
                         series: Dict[int, float],
                         legend: Optional[str] = None,
                         **kwargs):
    self._series.append((legend, kwargs, dict(sorted(series.items()))))

  def add_series_by_list(self,
                         series: Sequence[float],
                         legend: Optional[str] = None,
                         **kwargs):
    series = list(series)
    self.add_series_by_dict({i: v for i, v in enumerate(series)},
                            legend=legend,
                            **kwargs)

  def add_baseline(self, value: float, name: str):
    self._baselines[name] = value

  def plot(self, axes: Axes):
    super().plot(axes)

    for i, (legend, kwargs, series) in enumerate(self._series):
      xs, ys = [], []

      for index, value in series.items():
        xs.append(index)
        ys.append(value)

      axes.plot(xs, ys, label=legend, **kwargs)

    if self.show_legend:
      axes.legend(**self.legend_kwargs)
