import locale
import matplotlib.pyplot as plt
import numpy as np

class BasePlot:
  def __init__(self, output_path=None):
    plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
    plt.rcParams.update({'text.usetex':         True,
                         'text.latex.unicode':  True,
                         'font.family':         'lmodern',
                         'font.size':           10,
                         'axes.titlesize':      10,
                         'legend.fontsize':     10,
                         'legend.labelspacing': 0.2})

    fig = plt.figure()
    fig.set_size_inches(6, 3.8)

    self.do_plot(plt)

    plt.grid(True)
    plt.tight_layout()

    if output_path is None:
      plt.show()
    else:
      plt.savefig(output_path, dpi=1000, box_inches='tight')

  def do_plot(self, plt):
    raise NotImplementedError

class Plot(BasePlot):
  def __init__(self, stats, output_path=None):
    self.stats = stats
    BasePlot.__init__(self, output_path)

  def do_plot(self, plt):
    ind = np.arange(4)  # the x locations for the groups
    width = 0.25        # the width of the bars
    margin_left = (1 - 3 * width) / 2.0

    l = lambda x: x[self.get_stats_key()]
    naive = map(l, self.stats["naive"])
    simd  = map(l, self.stats["simd"])
    eigen = map(l, self.stats["eigen"])

    rects_naive = plt.bar(margin_left + ind, naive, width, color='r')
    rects_simd  = plt.bar(margin_left + ind + width, simd, width, color='y')
    rects_eigen = plt.bar(margin_left + ind + width * 2, eigen, width,
                          color='g')

    plt.title(self.get_title())
    plt.xticks(margin_left + ind + width * 1.5, ("-O0", "-O1", "-O2", "-O3"))
    plt.xlabel("Nivel de optimizaci\\'on")
    plt.ylabel("Tiempo (segundos)")
    plt.legend(("Naive", "SIMD", "Eigen"))

    axes = plt.axes()
    axes.set_ybound((0, max(naive + simd + eigen) * 1.05))

    def autolabel(rects):
      # attach some text labels
      for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2,
            height, "{:,}".format(int(height)), ha='center', va='bottom')

    autolabel(rects_naive)
    autolabel(rects_simd)
    autolabel(rects_eigen)

  def get_stats_key(self):
    raise NotImplementedError

  def get_title(self):
    raise NotImplementedError
