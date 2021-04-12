#include "cuda_config.h"

// get the maximum of two float numbers
__device__ void fatomicMax(float *addr, float value)
{
    float old = *addr, assumed;
    if(old >= value)
        return;
    do
    {
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(fmaxf(value, assumed)));
    }while(old!=assumed);
    return;
}

// get the minimum of two float numbers
__device__ void fatomicMin(float *addr, float value)
{
    float old = *addr, assumed;
    if(old <= value)
        return;
    do
    {
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(fminf(value, assumed)));
    }while(old!=assumed);
    return;
}

__global__ void visibility_kernel1(float *proj_points_, int *pixel_index_, float *front_image_, float *back_image_,
                                    int batch_size, int point_num, int height, int width, int visible_patch_width)
{
    CUDA_KERNEL_LOOP(index, batch_size * point_num)
    {
        int n = index / point_num;
        int pos = height * width;
        float *proj_point = proj_points_ + index * 3;
        int *pixel_index = pixel_index_ + index * 2;
        // float *visible_image = visible_image_ + n * pos;
        float *front_image = front_image_ + n * pos;
        float *back_image = back_image_ + n * pos;

        float project_x = proj_point[0];
        float project_y = proj_point[1];
        int id_x = max(min(int(project_x), width - 1), 0);
        int id_y = max(min(int(project_y), height - 1), 0);
        pixel_index[0] = id_x; // record i of pixel
        pixel_index[1] = id_y; // record j of pixel

        int init_x = id_x - (visible_patch_width - 1) / 2;
        int init_y = id_y - (visible_patch_width - 1) / 2;
        int image_index = 0;
        int current_x = 0;
        int current_y = 0;
        for (int j = 0; j < visible_patch_width; j++)
        {
            for (int i = 0; i < visible_patch_width; i++)
            {
                current_x = init_x + i;
                current_y = init_y + j;
                image_index = current_y * width + current_x;
                /*
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height)
                {
                    while(is_busy[image_index] == 1)
                    {
                        continue;
                    }
                    is_busy[image_index] = 1;
                    if (front_image[image_index] > proj_point[2])
                        front_image[image_index] = proj_point[2];
                    if (back_image[image_index] < proj_point[2])
                        back_image[image_index] = proj_point[2];
                    is_busy[image_index] = 0;
                }
                */
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height)
                {
                    fatomicMin(&front_image[image_index], proj_point[2]);
                    fatomicMax(&back_image[image_index], proj_point[2]);
                }
            }
        }
    }
}

__global__ void visibility_kernel2(float *proj_points_, int *pixel_index_, int *is_visible_,
                                    float *front_image_, float *back_image_,
                                    int batch_size, int point_num, int height, int width, float threshold)
{
    CUDA_KERNEL_LOOP(index, batch_size * point_num)
    {
        int n = index / point_num;
        int pos = height * width;
        float *proj_point = proj_points_ + index * 3;
        int *pixel_index = pixel_index_ + index * 2;
        int *is_visible = is_visible_ + index;
        // float *visible_image = visible_image_ + n * pos;
        float *front_image = front_image_ + n * pos;
        float *back_image = back_image_ + n * pos;

        int id_x = pixel_index[0];
        int id_y = pixel_index[1];
        /*
        if (proj_point[2] > (visible_image[id_y * width + id_x] + threshold))
        {
            is_visible[0] = 0;
        }
        */
        int image_index = id_y * width + id_x;
        if (proj_point[2] > ((front_image[image_index] + back_image[image_index]) * 0.5)
            && proj_point[2] > (front_image[image_index] + threshold))
        {
            atomicExch(&is_visible[0], 0);
            // is_visible[0] = 0;
        }
    }
}

__global__ void Rasterize_kernel1(float *proj_points_, int *pixel_index_, int *is_visible_, float *weight_points_, float *weight_image_,
                                    int batch_size, int point_num, int height, int width, int color_patch_width)
{
    CUDA_KERNEL_LOOP(index, batch_size * point_num)
    {
        int *is_visible = is_visible_ + index;
        if (is_visible[0] == 0)
        {
            return;
        }
        int n = index / point_num;
        // int point_id = index % point_num;
        int pos = height * width;
        int color_patch_size = color_patch_width * color_patch_width;
        float *proj_point = proj_points_ + index * 3; // n * 3 * point_num + 3 * point_id
        int *pixel_index = pixel_index_ + index * 2; // n * 2 * point_num + 2 * point_id
        float *weight_points = weight_points_ + index * color_patch_size; // n * point_num * color_patch_size + color_patch_size * point_id
        float *weight_image = weight_image_ + n * pos; // id_y * width + id_x
        
        float project_x = proj_point[0];
        float project_y = proj_point[1];
        int id_x = pixel_index[0]; // get i of pixel
        int id_y = pixel_index[1]; // get j of pixel

        int init_x = id_x - (color_patch_width - 1) / 2;
        int init_y = id_y - (color_patch_width - 1) / 2;
        float distance_divisor = float(((color_patch_width + 1) / 2) * ((color_patch_width + 1) / 2));
        int patch_index = 0; // index on patch
        int current_x = 0;
        int current_y = 0;
        float center_x = 0.0;
        float center_y = 0.0;
        float distance = 0.0;
        float gaussian_distance = 0.0;
        for (int j = 0; j < color_patch_width; j++)
        {
            for (int i = 0; i < color_patch_width; i++)
            {
                current_x = init_x + i;
                current_y = init_y + j;
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height)
                {
                    patch_index = j * color_patch_width + i;
                    center_x = float(2 * current_x + 1) / 2.0;
                    center_y = float(2 * current_y + 1) / 2.0;
                    distance = (project_x - center_x) * (project_x - center_x) + (project_y - center_y) * (project_y - center_y);
                    gaussian_distance = expf(-distance / distance_divisor);
                    weight_points[patch_index] = gaussian_distance;
                    atomicAdd(&weight_image[current_y * width + current_x], gaussian_distance);
                }
            }
        }
    }
}

__global__ void Rasterize_kernel2(float *proj_points_, int *pixel_index_, int *is_visible_, float *weight_points_, float *depth_image_,
                                    int batch_size, int point_num, int height, int width, int color_patch_width)
{
    CUDA_KERNEL_LOOP(index, batch_size * point_num)
    {
        int *is_visible = is_visible_ + index;
        if (is_visible[0] == 0)
        {
            return;
        }
        int n = index / point_num;
        // int point_id = index % point_num;
        int pos = height * width;
        int color_patch_size = color_patch_width * color_patch_width;
        float *proj_point = proj_points_ + index * 3; // n * 3 * point_num + 3 * point_id
        int *pixel_index = pixel_index_ + index * 2; // n * 2 * point_num + 2 * point_id
        float *weight_points = weight_points_ + index * color_patch_size; // n * point_num * color_patch_size + color_patch_size * point_id
        float *depth_image = depth_image_ + n * pos; // id_y * width + id_x
        
        int id_x = pixel_index[0]; // get i of pixel
        int id_y = pixel_index[1]; // get j of pixel

        int init_x = id_x - (color_patch_width - 1) / 2;
        int init_y = id_y - (color_patch_width - 1) / 2;
        int patch_index = 0; // index on patch
        int current_x = 0;
        int current_y = 0;
        float gaussian_distance = 0.0;
        for (int j = 0; j < color_patch_width; j++)
        {
            for (int i = 0; i < color_patch_width; i++)
            {
                current_x = init_x + i;
                current_y = init_y + j;
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height)
                {
                    patch_index = j * color_patch_width + i;
                    gaussian_distance = weight_points[patch_index];
                    atomicAdd(&depth_image[current_y * width + current_x], gaussian_distance * proj_point[2]);
                }
            }
        }
    }
}

__global__ void Rasterize_backward_kernel1(float *grad_depth_image_, float *grad_weight_image_,
                                            float *proj_points_, int *pixel_index_, int *is_visible_,
                                            float *grad_weight_points_,
                                            int batch_size, int point_num, int height, int width, int color_patch_width)
{
    CUDA_KERNEL_LOOP(index, batch_size * point_num)
    {
        int *is_visible = is_visible_ + index;
        if (is_visible[0] == 0)
        {
            return;
        }
        int n = index / point_num;
        // int point_id = index % point_num;
        int pos = height * width;
        int color_patch_size = color_patch_width * color_patch_width;
        float *proj_point = proj_points_ + index * 3;
        int *pixel_index = pixel_index_ + index * 2;
        float *grad_depth_image = grad_depth_image_ + n * pos;
        float *grad_weight_image = grad_weight_image_ + n * pos;
        float *grad_weight_point = grad_weight_points_ + index * color_patch_size;
        
        int id_x = pixel_index[0];
        int id_y = pixel_index[1];
        int init_x = id_x - (color_patch_width - 1) / 2;
        int init_y = id_y - (color_patch_width - 1) / 2;
        int patch_index = 0; // index on patch
        int image_index = 0; // index on image
        int current_x = 0;
        int current_y = 0;
        float grad_depth = 0.0;
        float grad_weight = 0.0;
        for (int j = 0; j < color_patch_width; j++)
        {
            for (int i = 0; i < color_patch_width; i++)
            {
                current_x = init_x + i;
                current_y = init_y + j;
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height)
                {
                    patch_index = j * color_patch_width + i;
                    image_index = current_y * width + current_x;
                    grad_depth = grad_depth_image[image_index] * proj_point[2];
                    grad_weight = grad_weight_image[image_index];
                    atomicAdd(&grad_weight_point[patch_index], grad_depth + grad_weight);
                }
            }
        }
    }
}

__global__ void Rasterize_backward_kernel2(float *grad_depth_image_,
                                            float *proj_points_, int *pixel_index_, int *is_visible_, float *weight_points_,
                                            float *grad_weight_points_, float *grad_proj_points_,
                                            int batch_size, int point_num, int height, int width, int color_patch_width)
{
    CUDA_KERNEL_LOOP(index, batch_size * point_num)
    {
        int *is_visible = is_visible_ + index;
        if (is_visible[0] == 0)
        {
            return;
        }
        int n = index / point_num;
        // int point_id = index % point_num;
        int pos = height * width;
        int color_patch_size = color_patch_width * color_patch_width;
        float *proj_point = proj_points_ + index * 3;
        int *pixel_index = pixel_index_ + index * 2;
        float *weight_point = weight_points_ + index * color_patch_size;
        float *grad_depth_image = grad_depth_image_ + n * pos;
        float *grad_weight_point = grad_weight_points_ + index * color_patch_size;
        float *grad_proj_point = grad_proj_points_ + index * 3;
        
        float project_x = proj_point[0];
        float project_y = proj_point[1];
        int id_x = pixel_index[0];
        int id_y = pixel_index[1];
        int init_x = id_x - (color_patch_width - 1) / 2;
        int init_y = id_y - (color_patch_width - 1) / 2;
        float distance_divisor = float(((color_patch_width + 1) / 2) * ((color_patch_width + 1) / 2));
        int patch_index = 0; // index on patch
        int image_index = 0; // index on image
        int current_x = 0;
        int current_y = 0;
        float center_x = 0.0;
        float center_y = 0.0;
        float grad_xy = 0.0;
        float grad_x = 0.0;
        float grad_y = 0.0;
        float grad_z = 0.0;
        for (int j = 0; j < color_patch_width; j++)
        {
            for (int i = 0; i < color_patch_width; i++)
            {
                current_x = init_x + i;
                current_y = init_y + j;
                if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height)
                {
                    patch_index = j * color_patch_width + i;
                    image_index = current_y * width + current_x;
                    center_x = float(2 * current_x + 1) / 2.0;
                    center_y = float(2 * current_y + 1) / 2.0;
                    grad_xy = -(grad_weight_point[patch_index] * weight_point[patch_index] * 2.0 / distance_divisor);
                    grad_x = grad_xy * (project_x - center_x);
                    grad_y = grad_xy * (project_y - center_y);
                    grad_z = grad_depth_image[image_index] * weight_point[patch_index];
                    atomicAdd(&grad_proj_point[0], grad_x);
                    atomicAdd(&grad_proj_point[1], grad_y);
                    atomicAdd(&grad_proj_point[2], grad_z);
                }
            }
        }
    }
}

void point_render_cuda_forward(float *proj_points, int *pixel_index, int *is_visible,
                                float *front_image, float *back_image,
                                float *weight_points, float *depth_image, float *weight_image,
                                int batch_size, int point_num, int height, int width,
                                int visible_patch_width, int color_patch_width, float threshold){
    visibility_kernel1<<<GET_BLOCKS(batch_size * point_num), CUDA_NUM_THREADS>>>(
        proj_points, pixel_index, front_image, back_image, batch_size, point_num, height, width, visible_patch_width);
    visibility_kernel2<<<GET_BLOCKS(batch_size * point_num), CUDA_NUM_THREADS>>>(
        proj_points, pixel_index, is_visible, front_image, back_image, batch_size, point_num, height, width, threshold);
    Rasterize_kernel1<<<GET_BLOCKS(batch_size * point_num), CUDA_NUM_THREADS>>>(
        proj_points, pixel_index, is_visible, weight_points, weight_image, batch_size, point_num, height, width, color_patch_width);
    Rasterize_kernel2<<<GET_BLOCKS(batch_size * point_num), CUDA_NUM_THREADS>>>(
        proj_points, pixel_index, is_visible, weight_points, depth_image, batch_size, point_num, height, width, color_patch_width);
}

void point_render_cuda_backward(float *grad_depth_image, float *grad_weight_image,
                                float *proj_points, int *pixel_index, int *is_visible, float *weight_points,
                                float *grad_weight_points, float *grad_proj_points,
                                int batch_size, int point_num, int height, int width, int color_patch_width){
    Rasterize_backward_kernel1<<<GET_BLOCKS(batch_size * point_num), CUDA_NUM_THREADS>>>(
        grad_depth_image, grad_weight_image,
        proj_points, pixel_index, is_visible,
        grad_weight_points,
        batch_size, point_num, height, width, color_patch_width);
    Rasterize_backward_kernel2<<<GET_BLOCKS(batch_size * point_num), CUDA_NUM_THREADS>>>(
        grad_depth_image,
        proj_points, pixel_index, is_visible, weight_points,
        grad_weight_points, grad_proj_points,
        batch_size, point_num, height, width, color_patch_width
    );
}