#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"

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
    std::cout << image.at<uint32_t>(0,0) << std::endl;

    //LandmarkCollection<cv::Vec2f> landmarks;
}

//    Mat image = cv::imread(imagefile.string());
//    LandmarkCollection<cv::Vec2f> landmarks;
//    try {
//        landmarks = read_pts_landmarks(landmarksfile.string());
//    }
//    catch (const std::runtime_error& e) {
//        cout << "Error reading the landmarks: " << e.what() << endl;
//        return EXIT_FAILURE;
//    }
//    morphablemodel::MorphableModel morphable_model;
//    try {
//        morphable_model = morphablemodel::load_model(modelfile.string());
//    }
//    catch (const std::runtime_error& e) {
//        cout << "Error loading the Morphable Model: " << e.what() << endl;
//        return EXIT_FAILURE;
//    }
//    core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);
//
//    // Draw the loaded landmarks:
//    Mat outimg = image.clone();
//    for (auto&& lm : landmarks) {
//        cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
//    }
//
//    // These will be the final 2D and 3D points used for the fitting:
//    vector<Vec4f> model_points; // the points in the 3D shape model
//    vector<int> vertex_indices; // their vertex indices
//    vector<Vec2f> image_points; // the corresponding 2D landmark points
//
//    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
//    for (int i = 0; i < landmarks.size(); ++i) {
//        auto converted_name = landmark_mapper.convert(landmarks[i].name);
//        if (!converted_name) { // no mapping defined for the current landmark
//            continue;
//        }
//        int vertex_idx = std::stoi(converted_name.get());
//        Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
//        model_points.emplace_back(vertex);
//        vertex_indices.emplace_back(vertex_idx);
//        image_points.emplace_back(landmarks[i].coordinates);
//    }
//
//    // Estimate the camera (pose) from the 2D - 3D point correspondences
//    fitting::ScaledOrthoProjectionParameters pose = fitting::estimate_orthographic_projection_linear(image_points, model_points, true, image.rows);
//    fitting::RenderingParameters rendering_params(pose, image.cols, image.rows);
//
//    // The 3D head pose can be recovered as follows:
//    float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
//    // and similarly for pitch and roll.
//
//    // Estimate the shape coefficients by fitting the shape to the landmarks:
//    Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
//    vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, affine_from_ortho, image_points, vertex_indices);
//
//    // Obtain the full mesh with the estimated coefficients:
//    render::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());
//
//    // Extract the texture from the image using given mesh and camera parameters:
//    Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);
//
//    // Save the mesh as textured obj:
//    outputfile += fs::path(".obj");
//    render::write_textured_obj(mesh, outputfile.string());
//
//    // And save the isomap:
//    outputfile.replace_extension(".isomap.png");
//    cv::imwrite(outputfile.string(), isomap);
//
//    cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfile.stem().stem() << "." << endl;
//
//    return EXIT_SUCCESS;
//}

/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 *
 * First, the 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper. Then, an affine camera matrix
 * is estimated, and then, using this camera matrix, the shape is fitted
 * to the landmarks.
 */
//int fit(int argc, char *argv[])
//{
//    // Load the image, landmarks, LandmarkMapper and the Morphable Model:
//    Mat image = cv::imread(imagefile.string());
//    LandmarkCollection<cv::Vec2f> landmarks;
//    try {
//        landmarks = read_pts_landmarks(landmarksfile.string());
//    }
//    catch (const std::runtime_error& e) {
//        cout << "Error reading the landmarks: " << e.what() << endl;
//        return EXIT_FAILURE;
//    }
//    morphablemodel::MorphableModel morphable_model;
//    try {
//        morphable_model = morphablemodel::load_model(modelfile.string());
//    }
//    catch (const std::runtime_error& e) {
//        cout << "Error loading the Morphable Model: " << e.what() << endl;
//        return EXIT_FAILURE;
//    }
//    core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);
//
//    // Draw the loaded landmarks:
//    Mat outimg = image.clone();
//    for (auto&& lm : landmarks) {
//        cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
//    }
//
//    // These will be the final 2D and 3D points used for the fitting:
//    vector<Vec4f> model_points; // the points in the 3D shape model
//    vector<int> vertex_indices; // their vertex indices
//    vector<Vec2f> image_points; // the corresponding 2D landmark points
//
//    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
//    for (int i = 0; i < landmarks.size(); ++i) {
//        auto converted_name = landmark_mapper.convert(landmarks[i].name);
//        if (!converted_name) { // no mapping defined for the current landmark
//            continue;
//        }
//        int vertex_idx = std::stoi(converted_name.get());
//        Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
//        model_points.emplace_back(vertex);
//        vertex_indices.emplace_back(vertex_idx);
//        image_points.emplace_back(landmarks[i].coordinates);
//    }
//
//    // Estimate the camera (pose) from the 2D - 3D point correspondences
//    fitting::ScaledOrthoProjectionParameters pose = fitting::estimate_orthographic_projection_linear(image_points, model_points, true, image.rows);
//    fitting::RenderingParameters rendering_params(pose, image.cols, image.rows);
//
//    // The 3D head pose can be recovered as follows:
//    float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
//    // and similarly for pitch and roll.
//
//    // Estimate the shape coefficients by fitting the shape to the landmarks:
//    Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
//    vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, affine_from_ortho, image_points, vertex_indices);
//
//    // Obtain the full mesh with the estimated coefficients:
//    render::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());
//
//    // Extract the texture from the image using given mesh and camera parameters:
//    Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);
//
//    // Save the mesh as textured obj:
//    outputfile += fs::path(".obj");
//    render::write_textured_obj(mesh, outputfile.string());
//
//    // And save the isomap:
//    outputfile.replace_extension(".isomap.png");
//    cv::imwrite(outputfile.string(), isomap);
//
//    cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfile.stem().stem() << "." << endl;
//
//    return EXIT_SUCCESS;
//}

PYBIND11_PLUGIN(fit) {
    py::module m("fit", "fit example");

    m.def("add", &add, "A function which adds two numbers");

    return m.ptr();
}
