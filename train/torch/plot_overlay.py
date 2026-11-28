import argparse, re, os
import matplotlib.pyplot as plt

FLOAT_REGEX = "([+-]?(\\d+([.]\\d*)?([eE][+-]?\\d+)?|[.]\\d+([eE][+-]?\\d+)?))"
INT_REGEX = "([+-]?\\d+)"

class LineDataParser:
    def __init__(self, line):
        self.line = line
        self.steps   = int(re.search(f"steps: {INT_REGEX}", line).group(1))
        self.samples = int(re.search(f"samples: {INT_REGEX}", line).group(1))
        self.policy  = float(re.search(f", prob loss: {FLOAT_REGEX}", line).group(1))
        self.wdl     = float(re.search(f", wdl loss: {FLOAT_REGEX}", line).group(1))

def moving_average(data, window):
    if window <= 1 or len(data) < window:
        return data
    return [sum(data[max(0, i - window + 1):i + 1]) / len(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]

def load_log(filename, cut_first, max_steps, x_axis):
    xx, policy_yy, wdl_yy = [], [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = LineDataParser(line)
            if d.steps <= cut_first:
                continue
            if max_steps is not None and d.steps > max_steps:
                continue
            xx.append(d.steps if x_axis == "steps" else d.samples)
            policy_yy.append(d.policy)
            wdl_yy.append(d.wdl)
    return xx, policy_yy, wdl_yy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs="+", required=True, help="Log files (e.g., exp8/training.log exp7/training.log)")
    parser.add_argument("-l", "--labels", nargs="+", default=None, help="Labels for each file (e.g., Exp8 Exp7)")
    parser.add_argument("--cut-first", type=int, default=5000, help="Cut first X steps (default: 5000)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps to plot (e.g., 60000)")
    parser.add_argument("--smooth", type=int, default=5, help="Moving average window (default: 5)")
    parser.add_argument("--x-axis", default="steps", choices=["steps", "samples"])
    parser.add_argument("-o", "--output", default="policy_wdl_overlay.png", help="Output image filename")
    args = parser.parse_args()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    # Style settings for each file (first = main, latest experiment; second = past experiment used for comparison)
    policy_styles = [
        {"color": "#1f77b4", "linestyle": "-",  "linewidth": 2.0}, # Blue, solid line (Exp 8)
        {"color": "#6baed6", "linestyle": "--", "linewidth": 1.6}, # Light blue, dashed line (Exp 7)
    ]
    wdl_styles = [
        {"color": "#d62728", "linestyle": "-",  "linewidth": 2.0}, # Red, solid line (Exp 8)
        {"color": "#ff9896", "linestyle": "--", "linewidth": 1.6}, # Salmon – dashed line (Exp 7)
    ]

    lines_for_legend = []

    for idx, filepath in enumerate(args.files):
        label_prefix = args.labels[idx] if (args.labels and idx < len(args.labels)) else os.path.basename(os.path.dirname(os.path.abspath(filepath))) or f"Run{idx+1}"
        xx, p_yy, w_yy = load_log(filepath, args.cut_first, args.max_steps, args.x_axis)
        if not xx:
            continue

        p_style = policy_styles[idx % len(policy_styles)]
        w_style = wdl_styles[idx % len(wdl_styles)]

        # 生データ（薄く表示）
        if args.smooth > 1:
            ax1.plot(xx, p_yy, color=p_style["color"], linewidth=0.7, alpha=0.2)
            ax2.plot(xx, w_yy, color=w_style["color"], linewidth=0.7, alpha=0.2)

        # 平滑化トレンド線
        p_smooth = moving_average(p_yy, args.smooth)
        w_smooth = moving_average(w_yy, args.smooth)

        l1, = ax1.plot(xx, p_smooth, label=f"{label_prefix} - Policy Loss", **p_style)
        l2, = ax2.plot(xx, w_smooth, label=f"{label_prefix} - WDL Loss", **w_style)
        lines_for_legend.extend([l1, l2])

    ax1.set_xlabel(args.x_axis.capitalize(), fontsize=11)
    ax1.set_ylabel("Policy (prob) Loss", color="#1f77b4", fontsize=11, fontweight="bold")
    ax2.set_ylabel("WDL Loss", color="#d62728", fontsize=11, fontweight="bold")

    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(linestyle="--", alpha=0.3)

    # Display the legend for all four lines together in the upper-right corner.
    labels = [l.get_label() for l in lines_for_legend]
    ax1.legend(lines_for_legend, labels, loc="upper right", framealpha=0.9)

    plt.title(f"Policy & WDL Loss Comparison (Steps > {args.cut_first}, MA={args.smooth})", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved overlay graph to: {args.output}")
    plt.show()

if __name__ == "__main__":
    main()

