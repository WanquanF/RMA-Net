#include "cuda_config.h"
#include <stdio.h>

__global__ void mask_kernel(float *proj_points_, float *mask_image_,
                            int batch_size, int point_num, int height, int width, int mask_patch_width)
{
    CUDA_KERNEL_LOOP(index, batch_size * point_num)
    {
        int n = index / point_num;
        int pos = height * width;
        float *proj_point = proj_points_ + index * 3;
        float *mask_image = mask_image_ + n * pos;

        float project_x = proj_point[0];
        float project_y = proj_point[1];
        int id_x = max(min(int(project_x), width - 1), 0);
        int id_y = max(min(int(project_y), height - 1), 0);

        int init_x = id_x - (mask_patch_width - 1) / 2;
        int init_y = id_y - (mask_patch_width - 1) / 2;
        int current_x = 0;
        int current_y = 0;
        for (int j = 0; j < mask_patch_width; j++)
        {
            for (int i = 0; i < mask_patch_width; i++)
            {
                current_x = init_x + i;
                current_y = init_y + j;
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height)
                {
                    mask_image[current_y * width + current_x] = -1.0;
                }
            }
        }
    }
}

__global__ void mask_backward_kernel(float *grad_mask_image_, float *proj_points_, float *grad_proj_points_,
                                        int batch_size, int point_num, int height, int width, float epsilon, float threshold)
{
    CUDA_KERNEL_LOOP(index, batch_size * height * width)
    {
        float *grad_mask_image = grad_mask_image_ + index;
        if (fabsf(grad_mask_image[0]) <= threshold)
        {
            return;
        }

        int pos = height * width;
        int n = index / pos;
        int pixel_id = index % pos;
        float *proj_point = proj_points_ + n * point_num * 3;
        float *grad_proj_point = grad_proj_points_ + n * point_num * 3;

        int current_x = pixel_id % width;
        int current_y = pixel_id / width;
        float center_x = float(2 * current_x + 1) / 2.0;
        float center_y = float(2 * current_y + 1) / 2.0;

        float gamma = 1000;
        float proj_x = 0.0;
        float proj_y = 0.0;
        float distance = 0.0;
        // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        // cudaArray *gaussian_distance;
        // cudaMallocArray(&gaussian_distance, &channelDesc, point_num);
        // float *gaussian_distance = (float *) malloc (point_num * sizeof(float));
        // float *gaussian_distance = new float[point_num];
        float gaussian_distance_sum = epsilon;
        for (int i = 0; i < point_num; i++)
        {
            proj_x = proj_point[3 * i];
            proj_y = proj_point[3 * i + 1];
            distance = (proj_x - center_x) * (proj_x - center_x) + (proj_y - center_y) * (proj_y - center_y);
            distance = expf(-distance / gamma);
            // gaussian_distance[i] = distance;
            gaussian_distance_sum += distance;
        }
        for (int i = 0; i < point_num; i++)
        {
            proj_x = proj_point[3 * i];
            proj_y = proj_point[3 * i + 1];
            distance = (proj_x - center_x) * (proj_x - center_x) + (proj_y - center_y) * (proj_y - center_y);
            distance = expf(-distance / gamma);
            atomicAdd(&grad_proj_point[3 * i + 2], grad_mask_image[0] * distance / gaussian_distance_sum);
        }
    }
}

void point_masker_cuda_forward(float *proj_points, float *mask_image,
                                int batch_size, int point_num, int height, int width, int mask_patch_width){
    mask_kernel<<<GET_BLOCKS(batch_size * point_num), CUDA_NUM_THREADS>>>(
        proj_points, mask_image, batch_size, point_num, height, width, mask_patch_width);
}

void point_masker_cuda_backward(float *grad_mask_image, float *proj_points, float *grad_proj_points,
                                int batch_size, int point_num, int height, int width, float epsilon, float threshold){
    mask_backward_kernel<<<GET_BLOCKS(batch_size * height * width), CUDA_NUM_THREADS>>>(
        grad_mask_image, proj_points, grad_proj_points, batch_size, point_num, height, width, epsilon, threshold);
}
