import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import blended_transform_factory

from imote.viz_tree.contributions import get_split_and_lin_contributions_test
from imote.nodes.leaf_node import LeafNode
from imote.viz_tree.viz_tree import VizTree

# Layout constants (in absolute figure units)
OUTER_MARGIN_HEIGHT = 0.2  # Top and bottom margin for outer plot
INNER_MARGIN_HEIGHT = 0.2  # Top and bottom margin for inner plots
DEFAULT_TICK_WIDTH = 0.65  # Space reserved for tick labels
DEFAULT_LABEL_WIDTH = 0.25  # Space reserved for axis labels
EXTRA_WIDTH = 0.15  # Extra horizontal spacing between plots

DOT_COLOR_UP = "red"
DOT_COLOR_DOWN = "blue"
DOT_LINE_WIDTH = 2
DOT_SIZE = 4


def predsplot2(viz_tree:VizTree, leaf_node:LeafNode, y_hat, n_max=5, fig_size=(10, 5),
              feature_names=None, all_feature_colors=None, display_type="histogram", truncate_total_pred=False,
              variable_tick_width=True, file_directory=None, highlight_x=None, staircase=False, short_labels=False):
    """
    Create a prediction plot for a vizualization of tree and a leaf node.

    Parameters
    ----------
    viz_tree
    leaf_node
    y_hat : array-like, shape (n_samples,)
        Predicted values for X
    n_max : int, default=5
        Maximum number of features to display
    fig_size : tuple, default=(12, 8)
        Figure size (width, height) in inches
    feature_names : List (n_features)
        Feature names of X
    all_feature_colors : List (n_features)
        Feature colors of X
    display_type : {"histogram", "density"}, default="histogram"
        Type of distribution display
    truncate_total_pred : bool, default=False
        If True, y-axis range based only on features contributions
    variable_tick_width : bool, default=True
        If True, optimizes layout based on actual tick label widths
    file_directory : str, optional
        If provided, saves figure to this path instead of displaying
    highlight_x : optional
        x values of point to highlight
    staircase : bool, default=False
        If True and highlight_x is not None, uses staircase view of highlighted point
    short_labels : bool, default=False
        If True, the right axis label is only "Total prediction", without the mean prediction, so it fits small plots
    """
    fig_width, fig_height = fig_size
    if fig_height < 3 and not short_labels:
        warnings.warn(
            "Plot height is small, the default titles will be cropped. "
            "Consider making the figure larger or higher."
        )

    highlight = highlight_x is not None

    # ============================================================================
    # STEP 1: Calculate contributions and identify features to plot
    # ============================================================================

    indices = leaf_node.indices
    coefficients = leaf_node.node_model.coefficients
    X = viz_tree.X_train[indices,:]
    all_split_contributions = viz_tree.split_contributions
    all_lin_contributions = viz_tree.linear_contributions
    split_contributions = all_split_contributions[indices,:]
    lin_contributions = all_lin_contributions[indices,:]
    center_predictions = np.mean(y_hat)
    n_features = X.shape[1]
    center_label = "mean prediction"

    centered_predictions = y_hat[indices] - center_predictions
    all_contributions = np.c_[lin_contributions, split_contributions]
    if highlight:
        highlight_x_split_contributions, highlight_x_lin_contributions = get_split_and_lin_contributions_test(viz_tree, highlight_x)
        highlight_x_all_contributions = np.r_[highlight_x_lin_contributions, highlight_x_split_contributions]
        highlight_x_sum_contributions = np.sum(highlight_x_all_contributions)

    # Rank features by variance in their contributions
    all_contribution_mean = np.mean(np.abs(all_contributions), axis=0)
    features_sorted_by_importance = np.argsort(all_contribution_mean)[::-1]

    # Select features with non-zero coefficients
    all_contribution_abs_max_per_feature = np.max(np.abs(all_contribution_mean), axis=0)
    abs_max_contribution = np.max(all_contribution_abs_max_per_feature)
    cutoff_zero_feature = abs_max_contribution * 0.01
    num_far_from_zero_features = np.sum(np.abs(all_contribution_mean) > cutoff_zero_feature)
    num_nonzero_features = np.sum(np.abs(all_contribution_mean) > 1e-14)
    n_max_plot = min(n_max, num_far_from_zero_features+1)
    num_features_to_plot = min(num_nonzero_features, n_max_plot)


    # Define feature labels
    if feature_names is None:
        lin_feature_labels = ["$X_{" + f"{i}" + "}$" for i in range(n_features)]
        split_feature_labels = ["Split\n$X_{" + f"{i}" + "}$" for i in range(n_features)]

    else:
        lin_feature_labels = feature_names
        split_feature_labels = [f"Split {feature_name}" for feature_name in feature_names]
    all_feature_labels = lin_feature_labels + split_feature_labels

    # Remove zero features and possibly combine last features
    if num_features_to_plot < num_nonzero_features:
        features_to_keep = features_sorted_by_importance[:n_max_plot - 1]
        features_to_combine = features_sorted_by_importance[n_max_plot - 1:num_nonzero_features]

        feature_data = np.empty((X.shape[0], n_max_plot))
        is_coefficients_positive = np.empty(n_max_plot)
        split_features = []
        for j in range(n_max_plot-1):
            i = features_to_keep[j]
            if i < n_features: #linear contribution
                feature_data[:, j] = X[:, i]
                is_coefficients_positive[j] = coefficients[i] > 0
            else: #split contribution
                feature_data[:, j] = all_contributions[:, i]
                is_coefficients_positive[j] = True
                split_features.append(j)
        feature_data[:, n_max_plot - 1] = np.sum(all_contributions[:, features_to_combine], axis=1)
        is_coefficients_positive[n_max_plot - 1] = True

        contributions = np.empty((X.shape[0], n_max_plot))
        contributions[:, :n_max_plot - 1] = all_contributions[:, features_to_keep]
        contributions[:, n_max_plot - 1] = np.sum(all_contributions[:, features_to_combine], axis=1)

        if np.var(feature_data[:, n_max_plot - 1]) < 1e-15:
            split_features.append(n_max_plot-1)

        features_labels = [all_feature_labels[i] for i in features_to_keep]
        features_labels.append("Remainder")

        if all_feature_colors is not None:
            feature_colors = [all_feature_colors[i%n_features] for i in features_to_keep]
            feature_colors.append("grey")

        if highlight:
            highlight_x_contributions = np.empty(n_max_plot)
            highlight_x_contributions[:n_max_plot - 1] = highlight_x_all_contributions[features_to_keep]
            highlight_x_contributions[n_max_plot - 1] = np.sum(highlight_x_all_contributions[features_to_combine])

            highlight_x_feature = np.empty(n_max_plot)
            for j in range(n_max_plot - 1):
                i = features_to_keep[j]
                if i < n_features:  # linear contribution
                    highlight_x_feature[j] = highlight_x[i]
                else:  # split contribution
                    highlight_x_feature[j] = highlight_x_all_contributions[i]
            highlight_x_feature[n_max_plot - 1] = highlight_x_contributions[n_max_plot - 1]
    else:
        features_to_keep = features_sorted_by_importance[:num_nonzero_features]
        feature_data = np.empty((X.shape[0], num_nonzero_features))
        is_coefficients_positive = np.empty(num_nonzero_features)
        split_features = []
        for j in range(num_nonzero_features):
            i = features_to_keep[j]
            if i < n_features: #linear contribution
                feature_data[:, j] = X[:, i]
                is_coefficients_positive[j] = coefficients[i] > 0
            else: #split contribution
                feature_data[:, j] = all_contributions[:, i]
                is_coefficients_positive[j] = True
                split_features.append(j)
        contributions = all_contributions[:, features_to_keep]
        features_labels = [all_feature_labels[i] for i in features_to_keep]
        if all_feature_colors is not None:
            feature_colors = [all_feature_colors[i%n_features] for i in features_to_keep]

        if highlight:
            highlight_x_contributions = highlight_x_all_contributions[features_to_keep]

            highlight_x_feature = np.empty(num_nonzero_features)
            for j in range(num_nonzero_features):
                i = features_to_keep[j]
                if i < n_features:  # linear contribution
                    highlight_x_feature[j] = highlight_x[i]
                else:  # split contribution
                    highlight_x_feature[j] = highlight_x_all_contributions[i]


    # Find range of contributions for each feature
    contribution_max_per_feature = np.max(contributions, axis=0)
    contribution_min_per_feature = np.min(contributions, axis=0)

    for i in split_features:
        if contribution_max_per_feature[i] > 0:
            contribution_min_per_feature[i] = 0
        else:
            contribution_max_per_feature[i] = 0

    # ============================================================================
    # STEP 2: Calculate initial layout dimensions
    # ============================================================================

    # Convert absolute sizes to figure-relative coordinates
    outer_margin_height = OUTER_MARGIN_HEIGHT / fig_height
    inner_margin_height = INNER_MARGIN_HEIGHT / fig_height
    tick_width = DEFAULT_TICK_WIDTH / fig_width
    label_width = DEFAULT_LABEL_WIDTH / fig_width
    extra_width = EXTRA_WIDTH / fig_width

    # Calculate available width for feature plots
    remaining_width = 1 - ((num_features_to_plot + 2) * tick_width + 2 * extra_width + 2 * label_width)
    feature_plot_width = remaining_width / (num_features_to_plot + 1)

    # Validate layout is feasible
    if feature_plot_width < 0:
        if variable_tick_width:
            # Will be recalculated after measuring actual tick widths
            feature_plot_width = 0.5 / fig_width
        else:
            raise ValueError(
                f"Plot is too narrow (feature plot width = {feature_plot_width}). "
                "Consider making the figure larger or wider."
            )
    elif not variable_tick_width and feature_plot_width < tick_width:
        warnings.warn("Plot width is small. Consider making the figure larger or wider.")

    # Determine predictions limits
    predictions_max_list = [np.max(contribution_max_per_feature)]
    predictions_min_list = [np.min(contribution_min_per_feature)]
    if not truncate_total_pred:
        predictions_max_list.append(np.max(centered_predictions))
        predictions_min_list.append(np.min(centered_predictions))
    if highlight:
        if staircase:
            cum_highlight_x_contribution = 0
            for i in range(1, num_features_to_plot):
                cum_highlight_x_contribution += highlight_x_contributions[i - 1]
                predictions_max_list.append(contribution_max_per_feature[i] + cum_highlight_x_contribution)
                predictions_min_list.append(contribution_min_per_feature[i] + cum_highlight_x_contribution)
        else:
            predictions_max_list.append(np.max(highlight_x_contributions))
            predictions_min_list.append(np.min(highlight_x_contributions))

        predictions_max_list.append(highlight_x_sum_contributions)
        predictions_min_list.append(highlight_x_sum_contributions)

    predictions_max = max(predictions_max_list)
    predictions_min = min(predictions_min_list)

    # Calculate plot area
    main_plot_left = label_width + tick_width
    main_plot_bottom = outer_margin_height
    main_plot_width = 1 - 2 * (tick_width + label_width)
    main_plot_height = 1 - 2 * outer_margin_height

    # Add inner margin in data coordinates
    inner_margin_height_data = (inner_margin_height *
                                (predictions_max - predictions_min) / (main_plot_height - 2 * inner_margin_height))
    predictions_max_with_margin = predictions_max + inner_margin_height_data
    predictions_min_with_margin = predictions_min - inner_margin_height_data

    # ============================================================================
    # STEP 3: Create main prediction distribution plot
    # ============================================================================

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Main plot showing prediction distribution
    ax_main = fig.add_axes([main_plot_left, main_plot_bottom, main_plot_width, main_plot_height])
    ax_main.invert_xaxis()

    # Plot distribution
    if display_type == "density":
        sns.kdeplot(y=centered_predictions, fill=True, ax=ax_main, color="grey")
    elif display_type == "histogram":
        sns.histplot(y=centered_predictions, fill=True, ax=ax_main, color="grey", linewidth=0)

    # Configure main plot appearance
    margin_multiplier = ((2 * extra_width + num_features_to_plot * (tick_width + feature_plot_width))
                         / feature_plot_width)
    ax_main.margins(x=margin_multiplier)
    ax_main.spines[['top', 'bottom', 'right']].set_visible(False)
    ax_main.tick_params(bottom=False, labelbottom=False)
    ax_main.ticklabel_format(scilimits=[-3, 4])
    ax_main.set_ylim([predictions_min_with_margin, predictions_max_with_margin])
    ax_main.set_xlabel(None)
    if highlight and staircase:
        ax_main.set_ylabel('Cumulative prediction contribution')
    else:
        ax_main.set_ylabel('Prediction contribution')

    # Right y-axis showing total predictions
    ax_right = ax_main.twinx()
    ax_right.ticklabel_format(scilimits=[-3, 4])
    ax_right.spines[['top', 'left', 'bottom']].set_visible(False)
    ax_right.set_ylim([
        predictions_min_with_margin + center_predictions,
        predictions_max_with_margin + center_predictions
    ])
    if short_labels:
        ax_right.set_ylabel("Total prediction")
    else:
        ax_right.set_ylabel(f"Total prediction (with {center_label} = {np.round(center_predictions, 2)})")

    if highlight:
        if highlight_x_sum_contributions < 0:
            dot_color = DOT_COLOR_DOWN
        else:
            dot_color = DOT_COLOR_UP

        ax_right.plot([0, 0], [center_predictions, highlight_x_sum_contributions + center_predictions], '-', color=dot_color,
                      linewidth=DOT_LINE_WIDTH, clip_on=False, zorder=10, solid_capstyle="butt")
        ax_right.plot(0, highlight_x_sum_contributions + center_predictions, 'o', color=dot_color, markersize=DOT_SIZE,
                      clip_on=False, zorder=12)
        if staircase:
            ax_right.plot(0, center_predictions, 'o', color='black', markersize=DOT_SIZE / 2, clip_on=False, zorder=11)

    # ============================================================================
    # STEP 4: Create feature distribution subplots (initial pass)
    # ============================================================================

    feature_axes = [None] * num_features_to_plot
    measured_tick_widths = np.empty(num_features_to_plot)
    subplot_bottom_positions = np.empty(num_features_to_plot)
    subplot_heights = np.empty(num_features_to_plot)

    subplot_height_offset = 0
    split_feature_width = (predictions_max_with_margin - predictions_min_with_margin)*0.02
    for i in range(num_features_to_plot):
        feature_values = feature_data[:, i]

        # Determine vertical position and height based on contribution range
        contribution_min = contribution_min_per_feature[i]
        contribution_max = contribution_max_per_feature[i]

        # Map contribution range to figure coordinates
        if highlight and staircase and i > 0:
            subplot_height_offset += highlight_x_contributions[i - 1]

        subplot_bottom_positions[i] = (
                ((contribution_min - predictions_min_with_margin) + subplot_height_offset) *
                main_plot_height / (predictions_max_with_margin - predictions_min_with_margin) + outer_margin_height
        )
        subplot_heights[i] = (
                (contribution_max - contribution_min) *
                main_plot_height / (predictions_max_with_margin - predictions_min_with_margin)
        )

        # Calculate horizontal position
        subplot_left = label_width + tick_width + extra_width + tick_width + i * (tick_width + feature_plot_width)

        # Create subplot
        feature_axes[i] = fig.add_axes([subplot_left, subplot_bottom_positions[i],
                                        feature_plot_width, subplot_heights[i]])

        # Configure subplot based on coefficient sign
        min_feature_value = feature_values.min()
        max_feature_value = feature_values.max()
        same_feature_value = min_feature_value == max_feature_value
        if same_feature_value & (min_feature_value <= 0): max_feature_value = 0
        elif same_feature_value & (max_feature_value > 0): min_feature_value = 0

        if is_coefficients_positive[i]:
            plot_color = "green"
            if not (features_labels[i] == "Remainder" or i in split_features):
                feature_axes[i].plot(0, max_feature_value, "^k", clip_on=False)
            feature_axes[i].set_ylim([min_feature_value, max_feature_value])
        else:
            plot_color = "firebrick"
            if not (features_labels[i] == "Remainder" or i in split_features):
                feature_axes[i].plot(0, max_feature_value, "vk", clip_on=False)
            feature_axes[i].set_ylim([max_feature_value, min_feature_value])
        if highlight:
            plot_color = "grey"
        if all_feature_colors is not None:
            plot_color = feature_colors[i]

        # Plot feature distribution
        if i in split_features:
            feature_axes[i].barh(y=feature_values[0], width=1, height=split_feature_width, color=plot_color)
        elif display_type == "density":
            sns.kdeplot(y=feature_values, fill=True, ax=feature_axes[i],
                        bw_adjust=1, color=plot_color, linewidth=0)
        elif display_type == "histogram":
            sns.histplot(y=feature_values, fill=True, ax=feature_axes[i], color=plot_color, linewidth=0)

        if highlight:
            highlight_x_start_value = (
                    (min_feature_value - max_feature_value)/(contribution_max - contribution_min)
                    *(contribution_min if is_coefficients_positive[i]> 0 else -contribution_max) + min_feature_value
            ) # Feature value where contribution is 0
            highlight_x_feature_value = highlight_x_feature[i]

            if bool(highlight_x_feature_value < highlight_x_start_value) ^ bool(is_coefficients_positive[i]):
                dot_color = DOT_COLOR_UP
            else:
                dot_color = DOT_COLOR_DOWN

            feature_axes[i].plot([0, 0], [highlight_x_start_value, highlight_x_feature_value], '-', color=dot_color,
                                 linewidth=DOT_LINE_WIDTH, clip_on=False, zorder=10, solid_capstyle="butt")
            feature_axes[i].plot(0, highlight_x_feature_value, 'o', color=dot_color, markersize=DOT_SIZE,
                                 clip_on=False, zorder=12)
            if staircase:
                feature_axes[i].plot(0, highlight_x_start_value, 'o', color='black', markersize=DOT_SIZE / 2,
                                     clip_on=False, zorder=11)

        # Configure subplot appearance
        feature_axes[i].spines[['top', 'right', 'bottom']].set_visible(False)
        if features_labels[i] == "Remainder" or i in split_features:
            feature_axes[i].set_yticks([])
        feature_axes[i].set_xlabel(features_labels[i])
        feature_axes[i].tick_params(bottom=False, labelbottom=False)
        feature_axes[i].ticklabel_format(scilimits=[-3, 4])

        # Measure actual tick label width
        if variable_tick_width:
            tick_labels = feature_axes[i].get_yticklabels()[1:-1]
            if len(tick_labels) < 1:
                tick_labels = feature_axes[i].get_yticklabels()

            tick_bboxes = [
                fig.transFigure.inverted().transform_bbox(label.get_window_extent())
                for label in tick_labels
            ]
            if len(tick_bboxes) == 0:
                measured_tick_widths[i] = extra_width / 2
            else:
                measured_tick_widths[i] = max([subplot_left - bbox.x0 for bbox in tick_bboxes])

    # ============================================================================
    # STEP 5: Optimize layout based on measured tick widths (if enabled)
    # ============================================================================

    mixed = blended_transform_factory(fig.transFigure, ax_main.transData)
    if variable_tick_width:
        # Measure tick widths for main axes
        main_tick_bboxes = [
            fig.transFigure.inverted().transform_bbox(label.get_window_extent())
            for label in ax_main.get_yticklabels()
        ]
        main_tick_width = max([main_plot_left - bbox.x0 for bbox in main_tick_bboxes])

        right_tick_bboxes = [
            fig.transFigure.inverted().transform_bbox(label.get_window_extent())
            for label in ax_right.get_yticklabels()
        ]
        right_tick_width = max([
            bbox.x1 - (main_plot_left + main_plot_width)
            for bbox in right_tick_bboxes
        ])

        # Recalculate main plot position
        optimized_main_plot_left = label_width + main_tick_width
        optimized_main_plot_width = (1 - 2 * label_width - main_tick_width - right_tick_width)
        ax_main.set_position([optimized_main_plot_left, main_plot_bottom, optimized_main_plot_width, main_plot_height])

        # Recalculate feature plot widths
        total_tick_width = np.sum(measured_tick_widths)
        optimized_remaining_width = (1 - total_tick_width - main_tick_width - right_tick_width -
                                     2 * extra_width - 2 * label_width)
        optimized_feature_width = optimized_remaining_width / (num_features_to_plot + 1)

        if optimized_feature_width < 0:
            raise ValueError(
                f"Plot is too narrow (optimized feature width = {optimized_feature_width}). "
                "Consider making the figure larger or wider."
            )

        # Reposition feature subplots
        for i in range(num_features_to_plot):
            optimized_subplot_left = (
                    label_width + main_tick_width + extra_width +
                    np.sum(measured_tick_widths[:i + 1]) +
                    i * optimized_feature_width
            )
            feature_axes[i].set_position([optimized_subplot_left, subplot_bottom_positions[i],
                                          optimized_feature_width, subplot_heights[i]])

            if highlight and staircase:
                if i == 0:
                    prev_optimized_subplot_left = optimized_subplot_left
                    staircase_line_height = highlight_x_contributions[i]
                else:
                    feature_axes[i - 1].add_artist(ConnectionPatch(
                        xyA=(prev_optimized_subplot_left, staircase_line_height), coordsA=mixed,
                        xyB=(optimized_subplot_left, staircase_line_height), coordsB=mixed,
                        capstyle="butt",
                        zorder=5
                    ))
                    prev_optimized_subplot_left = optimized_subplot_left
                    staircase_line_height += highlight_x_contributions[i]

                if i == num_features_to_plot - 1:
                    feature_axes[-1].add_artist(ConnectionPatch(
                        xyA=(prev_optimized_subplot_left, staircase_line_height), coordsA=mixed,
                        xyB=(
                        prev_optimized_subplot_left + optimized_feature_width + extra_width / 2, staircase_line_height),
                        coordsB=mixed,
                        capstyle="butt",
                        zorder=5
                    ))
                    ax_right.add_artist(ConnectionPatch(
                        xyA=(prev_optimized_subplot_left, staircase_line_height), coordsA=mixed,
                        xyB=(optimized_main_plot_left + optimized_main_plot_width, staircase_line_height),
                        coordsB=mixed,
                        capstyle="butt",
                        zorder=5
                    ))

        # Update main plot margins
        optimized_margin_multiplier = (2 * extra_width + total_tick_width + num_features_to_plot *
                                       optimized_feature_width) / optimized_feature_width
        ax_main.margins(x=optimized_margin_multiplier)

        # Horizontal reference line width
        if not highlight or not staircase:
            fig.add_artist(ConnectionPatch(
                xyA=(optimized_main_plot_left, 0), coordsA=mixed,
                xyB=(optimized_main_plot_left + optimized_main_plot_width, 0), coordsB=mixed,
                capstyle="butt"
            ))
    else:
        # Add horizontal reference line at y=0
        if not highlight or not staircase:
            fig.add_artist(ConnectionPatch(
                xyA=(main_plot_left, 0), coordsA=mixed,
                xyB=(main_plot_left + main_plot_width, 0), coordsB=mixed,
                capstyle="butt"
            ))

    # ============================================================================
    # STEP 6: Save or display figure
    # ============================================================================

    if file_directory is not None:
        fig.savefig(file_directory)
        plt.close(fig)
    else:
        plt.show()

    return None
