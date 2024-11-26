
import torch


def eval_depth(pred, target):
    """Evaluate how accurate a depth estimation was relative to 
    a labelled depth estimation target in different metrics. This
    evaluation function uses the following metrics to compare the
    depth estimation with the provided ground truth:
    - d1: The proportion of pixels in the estimation whose 
    depth value differs by more than 25% of the ground truth label.
    - d2: Similar to d1, but for pixels whose values differ by more than
    50%
    - d3: Similar to d1 and d2 but for a 75% difference.
    - absolute relative difference (abs_rel): The average difference
    between the ground truth pixel values in the depth image and their
    depth estimation. This is calculated as the mean of the absolute value
    difference between each pixel in the prediction and each pixel in
    the ground truth label.
    - square relative distance (sq_rel): Similar to the absolute relative 
    difference but the difference is squared rather than absolute-valued.
    - Root Mean Squared Error (RMSE): This metric is the quadratic mean 
    of the pixel-wise difference between the depth ground truth and the 
    estimation. This measures how far off the residual error of the estimation
    is from the original image. 
    - Log RMSE: Same as the RMSE but for the logarithmic pixel-wise difference.
    - SiLog loss: This is a global-local loss metric that measures both 
    the logarithmic pixel-wise loss and the average logarithmic difference 
    across all of the pixels between the depth ground truth and the estimation.
    Useful for accounting for both local and global error during training. 
    """
    # Raise an exception in the prediction and target are not equal in dimension
    # and size
    assert pred.shape == target.shape

    # Get the thresholds to be used for calculating the difference proportions
    thresh = torch.max((target / pred), (pred / target))

    # Get the pixel-wise distance threshold ratios between the ground truth
    # label and the depth estimation
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    # Calculate the pixel-wise difference and log difference 
    # between the depth ground truth and the estimation.
    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    # Calculate both the absolute and squared averages 
    # of the pixel-wise distances between the depth estimation
    # and original image.
    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    # Get the root-mean-squared error between
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    # Get both the logarithmic pixel-wise difference between 
    # the depth estimation and ground truth and the average 
    # of that logarithmic difference multiplied by a weighted constant.
    # This metric is useful for training a depth estimation model
    # as it accounts for both local and global error in the prediction
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    # Return all of the calculated metrics in a dictionary for easy use
    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


def eval_accuracy(predicted_depth, input_points, closer_point):
    """This function evaluates a depth estimation against two specific
    annoted pixels in the image as a ground truth where one of the pixels is
    closer. The two pixels are annotated by (height, width), with one of them
    being closer than the other. If the depth estimation labelled the closer pixel
    correclty, this function will return a 1, else it will return a zero."""
    # Get both of the annotated pixels in the image by width and height
    input_point_1, input_point_2 = input_points
    depth_point1 = predicted_depth[input_point_1[1],input_point_1[0]].item()
    depth_point2 = predicted_depth[input_point_2[1], input_point_2[0]].item()

    # Get which pixel the model predicted was closer in its estimation
    # or if there is a tie
    predicted_closer_point = 0
    if depth_point1 > depth_point2:
        predicted_closer_point = 0
    elif depth_point2 > depth_point1:
        predicted_closer_point = 1
    else:
        predicted_closer_point = 2

    # Return true if the estimated closer pixel is correct, 
    # else return false
    return predicted_closer_point == closer_point

