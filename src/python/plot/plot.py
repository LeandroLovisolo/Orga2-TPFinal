#!/usr/bin/env python2

# coding: utf-8

import argparse
from stats import load_stats
from base_plot import Plot

TOTAL_TRAINING_TIME_FILENAME = 'total-training-time.pdf'
AVG_EPOCH_TIME_FILENAME = "avg-epoch-time.pdf"

class TotalTrainingTimePlot(Plot):
  def __init__(self, stats, output_path):
    Plot.__init__(self, stats, output_path)

  def get_title(self):
    return "Tiempo total de entrenamiento por implementaci\\'on (100 \\'epocas)"

  def get_stats_key(self):
    return "total_training_time"

class AvgEpochTimePlot(Plot):
  def __init__(self, stats, output_path):
    Plot.__init__(self, stats, output_path)

  def get_title(self):
    return "Tiempo promedio por epoca de entrenamiento"

  def get_stats_key(self):
    return "avg_epoch_time"

def do_plot(plot, stats, output_path):
  if plot == "total-training-time":
    TotalTrainingTimePlot(stats, output_path)
  else:
    AvgEpochTimePlot(stats, output_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Plots total training times.")
  parser.add_argument(
      "-p", "--plot",
      help="Which plot to generate. Possible values: total-training-time, "
           "avg-epoch-time",
      dest="plot")
  parser.add_argument(
      "-d", "--directory",
      help="Path to directory containing stats files",
      dest="stats_directory_path")
  parser.add_argument(
      "-i", "--interactive",
      help="Display an interactive plot instead of saving as pdf.",
      dest="interactive", action="store_const", const=True, default=False)
  parser.add_argument(
      "-o", "--output",
      help="Output path",
      dest="output_path")
  args = parser.parse_args()
  assert(args.plot in ["total-training-time", "avg-epoch-time"])

  if args.interactive:
    output_path = None
  elif args.output_path:
    output_path = args.output_path
  else:
    if args.plot == "total-training-time":
      output_path = TOTAL_TRAINING_TIME_FILENAME
    else:
      output_path = AVG_EPOCH_TIME_FILENAME

  stats = load_stats(args.stats_directory_path)
  do_plot(args.plot, stats, output_path)
