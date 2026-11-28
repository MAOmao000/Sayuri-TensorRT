import argparse, re, os
import matplotlib.pyplot as plt

FLOAT_REGEX = "([+-]?(\\d+([.]\\d*)?([eE][+-]?\\d+)?|[.]\\d+([eE][+-]?\\d+)?))"
INT_REGEX = "([+-]?\\d+)"

X_AXIS_TYPE_LIST = ["steps", "samples"]
Y_AXIS_TYPE_LIST = ["all", "policy", "sprob", "optimistic", "ownership", "wdl", "q", "score"]
CUT_FIRST_DEFAULT = 3000

class LineDataParser:
    def __init__(self, line):
        self._match(line)

    def _single_match(self, fmt, as_type=None):
        val = re.search(fmt, self.line).group(1)
        if as_type:
            val = as_type(val)
        return val

    def _match(self, line):
        self.line = line
        self.steps      = self._single_match("steps: {}".format(INT_REGEX), int)
        self.samples    = self._single_match("samples: {}".format(INT_REGEX), int)
        self.speed      = self._single_match("speed: {}".format(FLOAT_REGEX), float)
        self.lr         = self._single_match("learning rate: {}".format(FLOAT_REGEX), float)
        self.batch_size = self._single_match("batch size: {}".format(INT_REGEX), int)

        self.count_dict = {"steps": self.steps, "samples": self.samples}
        self.loss_dict = {
            "all":        self._single_match("-> all loss: {}".format(FLOAT_REGEX), float),
            "policy":     self._single_match(", prob loss: {}".format(FLOAT_REGEX), float),
            "sprob":      self._single_match(", soft prob loss: {}".format(FLOAT_REGEX), float),
            "optimistic": self._single_match(", optimistic loss: {}".format(FLOAT_REGEX), float),
            "ownership":  self._single_match(", ownership loss: {}".format(FLOAT_REGEX), float),
            "wdl":        self._single_match(", wdl loss: {}".format(FLOAT_REGEX), float),
            "q":          self._single_match(", Q values loss: {}".format(FLOAT_REGEX), float),
            "score":      self._single_match(", scores loss: {}".format(FLOAT_REGEX), float),
        }

class GraphInfo:
    def __init__(self):
        self._data_list = list()

    def append(self, line):
        self._data_list.append(LineDataParser(line))

    def get_xy_plot(self, x_axis="steps", y_axis="all", cut_first=CUT_FIRST_DEFAULT, max_steps=None):
        xx, yy = list(), list()
        for line_data in self._data_list:
            if line_data.steps > cut_first:
                if max_steps is not None and line_data.steps > max_steps:
                    continue
                xx.append(line_data.count_dict[x_axis])
                yy.append(line_data.loss_dict[y_axis])
        return xx, yy

def moving_average(data, window):
    if window <= 1 or len(data) < window:
        return data
    return [sum(data[max(0, i - window + 1):i + 1]) / len(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]

def plot_process(args):
    graph_info_list = list()
    existing_names = set()
    for filename in args.filename:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        gi = GraphInfo()
        for l in lines:
            gi.append(l)
        # Using a folder name/file name format makes it easier to distinguish between files, even when they share the same name (such as `training.log`).
        parent = os.path.basename(os.path.dirname(os.path.abspath(filename)))
        label_name = f"{parent}/{os.path.basename(filename)}" if parent else os.path.basename(filename)
        graph_info_list.append((gi, label_name))

    plt.figure(figsize=(10, 5))
    for graph_info, file_tail in graph_info_list:
        xx, yy = graph_info.get_xy_plot(
            x_axis=args.x_axis,
            y_axis=args.y_axis,
            cut_first=args.cut_first,
            max_steps=args.max_steps
        )
        if len(xx) <= 1:
            continue
        while file_tail in existing_names:
            file_tail += "-1"
        existing_names.add(file_tail)

        if args.smooth > 1:
            line, = plt.plot(xx, yy, linestyle="-", linewidth=0.8, alpha=0.25)
            yy_smooth = moving_average(yy, args.smooth)
            plt.plot(xx, yy_smooth, linestyle="-", linewidth=2.0, color=line.get_color(), label=f"{file_tail} (MA{args.smooth})")
        else:
            plt.plot(xx, yy, linestyle="-", linewidth=1.2, label=file_tail)

    plt.ylabel(f"{args.y_axis} loss")
    plt.xlabel(args.x_axis)
    if args.log_scale:
        plt.xscale("log")
    if args.ylim:
        plt.ylim(args.ylim[0], args.ylim[1])
    plt.grid(linestyle="--", linewidth=1, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", nargs="+", required=True, type=str)
    parser.add_argument("--cut-first", default=CUT_FIRST_DEFAULT, type=int)
    parser.add_argument("--max-steps", default=None, type=int, help="Maximum steps to plot (e.g., 50000)")
    parser.add_argument("--smooth", default=5, type=int, help="Moving average window size (default: 5)")
    parser.add_argument("--ylim", nargs=2, default=None, type=float, help="Y-axis limits: min max")
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--y-axis", "--loss-type", default="all", choices=Y_AXIS_TYPE_LIST)
    parser.add_argument("--x-axis", default="steps", choices=X_AXIS_TYPE_LIST)
    args = parser.parse_args()
    plot_process(args)

