import random

import numpy as np


def convert_percentages_to_values(total, percentage1, percentage2, percentage3):
    # Calculate the scaling factor
    scaling_factor = total / sum([percentage1, percentage2, percentage3])

    # Calculate the values based on the scaled percentages and round them to the nearest integer
    val1 = round(percentage1 * scaling_factor)
    val2 = round(percentage2 * scaling_factor)
    val3 = round(percentage3 * scaling_factor)

    # Adjust the values to ensure their sum matches the total
    total_calculated = val1 + val2 + val3
    rounding_difference = total - total_calculated
    val1 += rounding_difference

    return val1, val2, val3


def compute_class_weights(true_values):
    class_counts = dict()
    for value in true_values:
        class_counts[value] = class_counts.get(value, 0) + 1

    total_samples = len(true_values)
    class_weights = {class_label: total_samples / (len(class_counts) * count) for class_label, count in
                     class_counts.items()}

    # Normalize the weights
    total_weight = sum(class_weights.values())
    class_weights_normalized = {class_label: weight / total_weight for class_label, weight in class_weights.items()}

    return np.array([class_weights_normalized[label] for label in true_values])


def generate_random_value_for_param(param):
    if 'min_value' in param and param['type'] == 'float':
        min_value = param['min_value']
        max_value = param['max_value']
        return random.uniform(min_value, max_value)
    elif 'min_value' in param:
        min_value = param['min_value']
        max_value = param['max_value']
        return random.randint(min_value, max_value)
    else:
        min_value = 0
        max_value = len(param) - 1
        return param[random.randint(min_value, max_value)]
