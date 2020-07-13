#include <iostream>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include "inference_framework.h"

using namespace dlib;
using namespace std;

int main() {
    std::string model_file = "/home/dell/dsj/alive_face_detection/model/graph_final.pb";
    int width = 160;
    int heigth = 160;
    std::string net_input = "input";
    std::string net_output = "embeddings";
    std::string pixel_normalization = "(0,1)";


    INFERENCE_FRAMEWORK face;
    face.init(model_file, width, heigth, pixel_normalization, net_input, net_output);
    cv::Mat out;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Can not open VideoCapture!" << endl;
    }
    cv::Mat Roi;
    cv::Mat face_image;
    cv::Mat temp;
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("/home/dell/dsj/alive_face_detection/model/landmarks_68.dat") >> pose_model;
    while (cv::waitKey(10) != 27) {
        cap >> temp;
        cv_image <bgr_pixel> cimg(temp);
        matrix <dlib::rgb_pixel> img;//Mat 转 matrix<dlib::rgb_pixel>
        assign_image(img, cv_image<rgb_pixel>(temp));

        std::vector <rectangle> faces = detector(cimg);
        full_object_detection shape_face;
        for (unsigned long i = 0; i < faces.size(); ++i) {
            shape_face = pose_model(cimg, faces[i]);
            matrix <rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape_face, 160, 0.25), face_chip);//需要检测时再进行图片矫正
            face_image = dlib::toMat(face_chip);//face_image为dlib矫正后输出人脸图像
            out = face.infer(face_image);

            cout << "net out:" << out << endl;
            cout << "*****************************************" << endl;

        }
    }
}