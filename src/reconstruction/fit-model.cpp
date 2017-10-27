#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"
//#include "eos/pybind11_glm.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace eos; 
namespace po = boost::program_options;
namespace fs = boost::filesystem;

using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/**
 * Print colored image in grayscale
 */
void print_image(uint32_t *image, py::buffer_info *image_info) {
    size_t width = image_info->shape[1];
    size_t height = image_info->shape[0];
    size_t color = image_info->shape[2];

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            auto B = image[y * width + x + 0];
            auto G = image[y * width + x + 1];
            auto R = image[y * width + x + 2];

            std::cout << (int)((B + G + R) / 3.0) << " ";
        }

        std::cout << std::endl;
    }
}

float add(py::array_t<uint32_t> image_input,
          py::array_t<float> input_points) {
    auto image_info = image_input.request();
    auto points_info = input_points.request();

    uint32_t *image = reinterpret_cast<uint32_t *>(image_info.ptr);
    float *points = reinterpret_cast<float *>(points_info.ptr);

    //std::cout << image_info.shape << std::endl;
    std::cout << image_info.itemsize << std::endl;
    std::cout << image_info.ndim << std::endl;
    std::cout << image_info.format << std::endl;

    std::cout << image_info.shape[0] << std::endl;
    std::cout << image_info.shape[1] << std::endl;

    print_image(image, &image_info);

    return 1.0;
}


void fit(Mat image, Vec2f landmarks) {
    // std::cout << image.at<uint32_t>(0,0) << std::endl;
    std::cout << landmarks << std::endl;
    return;
    LandmarkCollection<cv::Vec2f> landmarkCollection;
    morphablemodel::MorphableModel morphable_model;

    try {
        morphable_model = morphablemodel::load_model("/usr/local/eos/share/sfm_shape_3448.bin");
    }
    catch (const std::runtime_error& e) {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return;
    }

    core::LandmarkMapper landmark_mapper = core::LandmarkMapper("/usr/local/eos/share/bug2did.txt");

    // Draw the loaded landmarks:
    Mat outimg = image.clone();
    for (auto&& lm : landmarkCollection) {
        cv::rectangle(
            outimg,
            cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
            cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f),
            { 255, 0, 0 }
        );
    }

    // These will be the final 2D and 3D points used for the fitting:
    vector<Vec4f> model_points; // the points in the 3D shape model
    vector<int> vertex_indices; // their vertex indices
    vector<Vec2f> image_points; // the corresponding 2D landmark points

    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
    for (int i = 0; i < landmarkCollection.size(); ++i) {
        auto converted_name = landmark_mapper.convert(landmarkCollection[i].name);
        if (!converted_name) { // no mapping defined for the current landmark
            continue;
        }
        int vertex_idx = std::stoi(converted_name.get());
        Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
        model_points.emplace_back(vertex);
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarkCollection[i].coordinates);
    }

    // Estimate the camera (pose) from the 2D - 3D point correspondences
    fitting::ScaledOrthoProjectionParameters pose = fitting::estimate_orthographic_projection_linear(image_points, model_points, true, image.rows);
    fitting::RenderingParameters rendering_params(pose, image.cols, image.rows);

    // The 3D head pose can be recovered as follows:
    float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
    // and similarly for pitch and roll.

    // Estimate the shape coefficients by fitting the shape to the landmarks:
    Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
    vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, affine_from_ortho, image_points, vertex_indices);

    // Obtain the full mesh with the estimated coefficients:
    render::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());

    // Extract the texture from the image using given mesh and camera parameters:
    Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);

    // Save the mesh as textured obj:
    render::write_textured_obj(mesh, "/data/fit-model-test.obj");

    // And save the isomap:
    cv::imwrite("/data/fit-model-test.isomap.png", isomap);
}


PYBIND11_PLUGIN(fit) {
    py::module m("fit", "fit example");

    /**
    * General bindings, for OpenCV vector types and cv::Mat:
    *  - cv::Vec2f
    *  - cv::Vec4f
    *  - cv::Mat (only 1-channel matrices and only conversion of CV_32F C++ matrices to Python, and conversion of CV_32FC1 and CV_64FC1 matrices from Python to C++)
    */
    //py::class_<cv::Vec2f>(fit, "Vec2f", "Wrapper for OpenCV's cv::Vec2f type.")
    //.def("__init__", [](cv::Vec2f& vec, py::buffer b) {
    //    py::buffer_info info = b.request();

    //    if (info.ndim != 1)
    //        throw std::runtime_error("Buffer ndim is " + std::to_string(info.ndim) + ", please hand a buffer with dimension == 1 to create a Vec2f.");
    //    if (info.strides.size() != 1)
    //        throw std::runtime_error("strides.size() is " + std::to_string(info.strides.size()) + ", please hand a buffer with strides.size() == 1 to create a Vec2f.");
    //    // Todo: Should add a check that checks for default stride sizes, everything else would not work yet I think.
    //    if (info.shape.size() != 1)
    //        throw std::runtime_error("shape.size() is " + std::to_string(info.shape.size()) + ", please hand a buffer with shape dimension == 1 to create a Vec2f.");
    //    if (info.shape[0] != 2)
    //        throw std::runtime_error("shape[0] is " + std::to_string(info.shape[0]) + ", please hand a buffer with 2 entries to create a Vec2f.");

    //    if (info.format == py::format_descriptor<float>::format()) {
    //        cv::Mat temp(1, 2, CV_32FC1, info.ptr);
    //        // std::cout << temp << std::endl;
    //        new (&vec) cv::Vec2f(temp);
    //    } else {
    //        throw std::runtime_error("Not given a buffer of type float - please hand a buffer of type float to create a Vec2f.");
    //    }
    //})
    //.def_buffer([](cv::Vec2f& vec) -> py::buffer_info {
    //return py::buffer_info(
    //    &vec.val,                               /* Pointer to buffer */
    //    sizeof(float),                          /* Size of one scalar */
    //    py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
    //    2,                                      /* Number of dimensions */
    //    { vec.rows, vec.cols },                 /* Buffer dimensions */
    //    { sizeof(float),             /* Strides (in bytes) for each index */
    //    sizeof(float) }             //=> both sizeof(float), since the data is hold in an array, i.e. contiguous memory */
    //);
    //});

    m.def("add", &add, "A function which adds two numbers");
    m.def("fit", &fit, "Fit");

    return m.ptr();
}
