#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>
#include "cuda_config.h"

#define gpuErrchk(ans)                    \
    {                                     \
    gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// C++ interface
#define CHECK_CUDA(x) \
    AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void point_render_cuda_forward(float *proj_points, int *pixel_index, int *is_visible,
                                float *front_image, float *back_image,
                                float *weight_points, float *depth_image, float *weight_image,
                                int batch_size, int point_num, int height, int width,
                                int visible_patch_width, int color_patch_width, float threshold);

void point_render_cuda_backward(float *grad_depth_image, float *grad_weight_image,
                                float *proj_points, int *pixel_index, int *is_visible, float *weight_points,
                                float *grad_weight_points, float *grad_proj_points,
                                int batch_size, int point_num, int height, int width, int color_patch_width);

std::vector<at::Tensor> point_render_forward(at::Tensor proj_points, int image_height_, int image_width_,
                                                int visible_patch_width_, int color_patch_width_, float epsilon_, float threshold_)
{
    CHECK_INPUT(proj_points);

    int batch_size = proj_points.size(0);
    int point_num = proj_points.size(1);
    int height = image_height_;
    int width = image_width_;
    int visible_patch_width = visible_patch_width_;
    int color_patch_width = color_patch_width_;
    float epsilon = epsilon_;
    float threshold = threshold_;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto options_int_nograd = torch::TensorOptions()
                                .dtype(torch::kInt32)
                                .layout(proj_points.layout())
                                .device(proj_points.device())
                                .requires_grad(false);
    auto options_float_ifgrad = torch::TensorOptions()
                                .dtype(proj_points.dtype())
                                .layout(proj_points.layout())
                                .device(proj_points.device())
                                .requires_grad(proj_points.requires_grad());
    auto options_float_grad = torch::TensorOptions()
                                .dtype(proj_points.dtype())
                                .layout(proj_points.layout())
                                .device(proj_points.device())
                                .requires_grad(true);
    auto options_float_nograd = torch::TensorOptions()
                                .dtype(proj_points.dtype())
                                .layout(proj_points.layout())
                                .device(proj_points.device())
                                .requires_grad(false);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    auto pixel_index = -torch::ones({batch_size, 2*point_num}, options_int_nograd);
    auto is_visible = torch::ones({batch_size, point_num}, options_int_nograd);
    // auto is_busy = torch::zeros({batch_size, height, width, 1}, options_int_nograd);
    // auto visible_image = 1000.0 * torch::ones({batch_size, height, width, 1}, options_float_nograd);
    auto front_image = torch::zeros({batch_size, height, width, 1}, options_float_nograd); // z-value max is 0
    auto back_image = -2.0 * torch::ones({batch_size, height, width, 1}, options_float_nograd); // z-value min is -2
    auto weight_points = torch::zeros({batch_size, point_num, color_patch_width*color_patch_width}, options_float_nograd);
    auto depth_image = torch::zeros({batch_size, height, width, 1}, options_float_ifgrad);
    auto weight_image = epsilon * torch::ones({batch_size, height, width, 1}, options_float_ifgrad);

    point_render_cuda_forward(
        proj_points.data<float>(), pixel_index.data<int>(), is_visible.data<int>(),
        front_image.data<float>(), back_image.data<float>(),
        weight_points.data<float>(), depth_image.data<float>(), weight_image.data<float>(),
        batch_size, point_num, height, width, visible_patch_width, color_patch_width, threshold);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {pixel_index, is_visible, weight_points, depth_image, weight_image};
}

std::vector<at::Tensor> point_render_backward(at::Tensor grad_depth_image, at::Tensor grad_weight_image, at::Tensor proj_points,
                                                at::Tensor pixel_index, at::Tensor is_visible, at::Tensor weight_points)
{
    CHECK_INPUT(grad_depth_image);
    CHECK_INPUT(grad_weight_image);
    CHECK_INPUT(pixel_index);
    CHECK_INPUT(is_visible);
    CHECK_INPUT(weight_points);

    int batch_size = proj_points.size(0);
    int point_num = proj_points.size(1);
    int height = grad_depth_image.size(1);
    int width = grad_depth_image.size(2);
    int color_patch_width = int(sqrt(weight_points.size(2)));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto grad_proj_points = at::zeros_like(proj_points);
    auto grad_weight_points = at::zeros_like(weight_points);

    point_render_cuda_backward(
        grad_depth_image.data<float>(), grad_weight_image.data<float>(),
        proj_points.data<float>(), pixel_index.data<int>(), is_visible.data<int>(), weight_points.data<float>(),
        grad_weight_points.data<float>(), grad_proj_points.data<float>(),
        batch_size, point_num, height, width, color_patch_width
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {grad_proj_points};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &point_render_forward, "Point-Rendering forward (CUDA)");
    m.def("backward", &point_render_backward, "Point-Rendering backward (CUDA)");
}
