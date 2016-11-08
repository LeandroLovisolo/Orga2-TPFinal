import os

NAIVE_O0_PATH = "naive-O0.txt"
NAIVE_O1_PATH = "naive-O1.txt"
NAIVE_O2_PATH = "naive-O2.txt"
NAIVE_O3_PATH = "naive-O3.txt"

SIMD_O0_PATH = "simd-O0.txt"
SIMD_O1_PATH = "simd-O1.txt"
SIMD_O2_PATH = "simd-O2.txt"
SIMD_O3_PATH = "simd-O3.txt"

EIGEN_O0_PATH = "eigen-O0.txt"
EIGEN_O1_PATH = "eigen-O1.txt"
EIGEN_O2_PATH = "eigen-O2.txt"
EIGEN_O3_PATH = "eigen-O3.txt"

def load_stats(directory_path):
  stats = {}
  files = {
    "naive": [NAIVE_O0_PATH, NAIVE_O1_PATH, NAIVE_O2_PATH, NAIVE_O3_PATH],
    "simd":  [SIMD_O0_PATH,  SIMD_O1_PATH,  SIMD_O2_PATH,  SIMD_O3_PATH],
    "eigen": [EIGEN_O0_PATH, EIGEN_O1_PATH, EIGEN_O2_PATH, EIGEN_O3_PATH]
  }
  for impl in files:
    stats[impl] = []
    for path in files[impl]:
      stats[impl].append(parse_stats_file_(directory_path + os.sep + path))
  return stats

def parse_stats_file_(path):
  stats = {}
  with open(path) as f:
    lines = f.readlines()
  for line in lines:
    label, value = line.split(":")
    label = label.strip()
    value = float(value)
    stats[label] = value
  return stats
