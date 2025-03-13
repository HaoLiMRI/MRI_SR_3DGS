import torch
# import torch.nn.functional as F
import math

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion (x, y, z, w) to rotation matrix.
    q: Tensor of shape (batch_size, 4)
    """
    batch_size, height, width, depth = q.shape[:4]
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).view(batch_size, height, width, depth, 3, 3)
    
    return R

def gaussian_distribution(x, mu, sigma, rotation): #voxel_center, grid+anchor, sigma, rotation
    """
    Compute the Gaussian distribution values for a given point.
    """
#    if torch.isnan(x).any():
#        print("NaN detected in x!")
    x = x.unsqueeze(1).unsqueeze(2).unsqueeze(3)   # [5, 1, 1, 1, 3] 
    x_rotated = torch.matmul(rotation, (x.expand(mu.shape) - mu).unsqueeze(-1)).squeeze(-1)
#    if torch.isnan(x_rotated).any():
#        print("NaN detected in x_rotated!")
    exponent = -0.5 * torch.sum((x_rotated / sigma) ** 2, dim=-1)
#    if torch.isnan(exponent).any():
#        print("NaN detected in exponent!")
    distribution = torch.exp(exponent) / (torch.prod(sigma, dim=-1) * (2 * math.pi) ** 1.5)
#    if torch.isnan(distribution).any():
#        print("NaN detected in distribution!")
    return distribution

def high_res_image_rendering(matrix):
    """
    Reconstruct high-resolution 3D image from network output matrix.

    Args:
        matrix: Tensor of shape (b, 11, h, w, d), network output matrix.
        h, w, d: Dimensions of the high-resolution 3D image.

    Returns:
        high_res_image: Tensor of shape (b, h, w, d), reconstructed image.
    """
    b, _, h, w, d = matrix.shape
    voxel_size = 1.0  # Assuming each voxel size is 1x1x1

    # Extract components from matrix
    I = matrix[:, 0]  # Signal intensity
    quaternion = matrix[:, 1:5].permute(0,2,3,4,1)   # Rotation quaternion
    sigma = matrix[:, 5:8].permute(0,2,3,4,1)   # Gaussian variances
    anchor = matrix[:, 8:11].permute(0,2,3,4,1)   # Anchor positions

    # Compute rotation matrix from quaternion
#    rotation = quaternion_to_rotation_matrix(quaternion.view(-1, 4)).view(b, h, w, d, 3, 3)
    rotation = quaternion_to_rotation_matrix(quaternion).view(b, h, w, d, 3, 3)

    # Prepare voxel grid
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0, h - 1, h),
        torch.linspace(0, w - 1, w),
        torch.linspace(0, d - 1, d),
        indexing='ij'
    ), dim=-1).to(matrix.device)  # (h, w, d, 3)
    grid = grid.unsqueeze(0).expand(b, h, w, d, 3) + voxel_size/2  # Broadcast to batch

    # Initialize high-resolution image
    high_res_image = torch.zeros((b, h, w, d), device=matrix.device)

    for i in range(h):
        for j in range(w):
            for k in range(d):
#                voxel_center = grid[:, i, j, k]  # Center of voxel
                gaussian_values = gaussian_distribution(grid[:, i, j, k], grid+anchor, sigma, rotation)
                high_res_image[:, i, j, k] = torch.sum(I * gaussian_values, dim = (1, 2, 3))
    
#    if torch.isnan(high_res_image).any():
#        print("NaN detected in high_res_image!")
#    print("high_res_image requires_grad:", high_res_image.requires_grad)
#    print("high_res_image.grad_fn:", high_res_image.grad_fn)

    return high_res_image

