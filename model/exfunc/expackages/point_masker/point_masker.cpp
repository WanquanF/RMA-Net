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

void point_masker_cuda_forward(float *proj_points, float *mask_image,
                                int batch_size, int point_num, int height, int width, int mask_patch_width);

void point_masker_cuda_backward(float *grad_mask_image, float *proj_points, float *grad_proj_points,
                                int batch_size, int point_num, int height, int width, float epsilon, float threshold);

std::vector<at::Tensor> point_masker_forward(at::Tensor proj_points, int image_height_, int image_width_, int mask_patch_width_)
{
    CHECK_INPUT(proj_points);
    
    int batch_size = proj_points.size(0);
    int point_num = proj_points.size(1);
    int height = image_height_;
    int width = image_width_;
    int mask_patch_width = mask_patch_width_;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto options_float_ifgrad = torch::TensorOptions()
                                .dtype(proj_points.dtype())
                                .layout(proj_points.layout())
                                .device(proj_points.device())
                                .requires_grad(proj_points.requires_grad());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    auto mask_image = torch::zeros({batch_size, height, width, 1}, options_float_ifgrad);

    point_masker_cuda_forward(proj_points.data<float>(), mask_image.data<float>(),
        batch_size, point_num, height, width, mask_patch_width
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {mask_image};
}

std::vector<at::Tensor> point_masker_backward(at::Tensor grad_mask_image, at::Tensor proj_points,
                                                float epsilon_, float threshold_)
{
    CHECK_INPUT(grad_mask_image);

    int batch_size = proj_points.size(0);
    int point_num = proj_points.size(1);
    int height = grad_mask_image.size(1);
    int width = grad_mask_image.size(2);
    float epsilon = epsilon_;
    float threshold = threshold_;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto grad_proj_points = at::zeros_like(proj_points);

    point_masker_cuda_backward(grad_mask_image.data<float>(), proj_points.data<float>(),
        grad_proj_points.data<float>(),
        batch_size, point_num, height, width, epsilon, threshold
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {grad_proj_points};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &point_masker_forward, "Point-Maskering forward (CUDA)");
    m.def("backward", &point_masker_backward, "Point-Maskering backward (CUDA)");
}