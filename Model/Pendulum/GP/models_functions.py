import numpy as np


def check_settling_time(prediction, goal):
    # Ensure both arrays have at least 20 entries
    if len(prediction) < 20 or len(goal) < 20:
        raise ValueError("Both arrays should have at least 20 entries.")

    # Take the last 20 entries from both arrays
    last_20_prediction = prediction[-20:]
    last_20_goal = goal[-20:]

    # Calculate the maximum allowed average error (2%)
    max_average_error = 0.02

    # Check if the average error across all dimensions is within the 2% error band
    for val1, val2 in zip(last_20_prediction, last_20_goal):

        errors = np.abs((val1 - val2) / val2)  # Calculate the percentage error for each dimension
        avg_error = np.mean(errors)  # Calculate the average error across all dimensions
        if avg_error > max_average_error:
            return False  # Average error exceeds the 2% limit

    return True  # Average error is within the 2% limit for all 20 entries