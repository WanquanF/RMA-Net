import torch
import point_render
import point_masker

"""
    Define a renderer tool.
    For each time, input a proj_points (B, N_V, 3) and threshold (float), output an depth_image (B, h, w, 1) and weight image (B, h, w, 1):
        depth_image / weight_image = render_image
    
    image_height: the height of image
    image_width: the width of image
    visible_patch_width: the width of patch which define if a point is visible.
    color_path_width: the width of patch which define the region a point affects.
    epsilon: avoid the divisor == 0
    
    proj_points: (B, N_V, 3). N_V is the number of points. (x, y) must on the image, z must under 0.0 (z smaller, more closer to camera).
    threshold: the threshold which decides if the point is visible.
"""

class Masker_Point(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj_points, image_height, image_width, mask_patch_width, epsilon, threshold):
        ctx.threshold = threshold
        ctx.epsilon = epsilon
        mask_image = point_masker.forward(proj_points, image_height, image_width, mask_patch_width)[0]
        ctx.save_for_backward(proj_points)
        return mask_image

    @staticmethod
    def backward(ctx, grad_mask_image):
        proj_points = ctx.saved_tensors[0]
        grad_proj_points = point_masker.backward(grad_mask_image, proj_points, ctx.epsilon, ctx.threshold)[0]
        return grad_proj_points, None, None, None, None, None

class Masker(torch.nn.Module):
    def __init__(self, image_height=256, image_width=256, mask_patch_width=5, epsilon=1e-5):
        super(Masker, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.mask_patch_width = mask_patch_width
        self.epsilon = epsilon # avoid division by zero

    def forward(self, proj_points, threshold):
        return Masker_Point.apply(proj_points, self.image_height, self.image_width, self.mask_patch_width, self.epsilon, threshold)
