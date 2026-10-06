from matplotlib.colors import LinearSegmentedColormap

# Blue -> red color map sampled from SHAP's `shap.plots.colors.red_blue` (MIT license),
# so IMoTE doesn't need shap as a dependency just for this color map.
red_blue = LinearSegmentedColormap.from_list("red_blue", [
    "#008bfb", "#007df5", "#236cea", "#625adb", "#8443c6", "#9c23ad",
    "#bb00a0", "#d5008f", "#e9007c", "#f90067", "#ff0051",
])
red_blue.set_bad("#848484")
red_blue.set_over("#848484")
red_blue.set_under("#848484")
