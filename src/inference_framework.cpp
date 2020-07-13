//
// Created by dell on 20-7-13.
//

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "inference_framework.h"


void INFERENCE_FRAMEWORK::init(std::string model_file, int width, int heigth, std::string pixel_normalization,
                               std::string net_input, std::string net_output){
    std::cout << "INFERENCE_FRAMEWORK::init" << std::endl;

    wid = width;
    hig = heigth;
    model = model_file;
    net_input_name = net_input;
    net_output_name = net_output;
    pixel_norm = pixel_normalization;
    try {
        net = cv::dnn::readNetFromTensorflow(model);
        for (auto n: net.getLayerNames()) {
            std::cout << n << std::endl;
        }
    }
    catch (std::exception &e) {
        std::cout << "Can not find model!" << e.what() << std::endl;
        return;
    }
    if (net.empty()) {
        printf("Read tf model data failure...\n");
        return;
    }
    std::cout << "INFERENCE_FRAMEWORK::init ok" << std::endl;
}

cv::Mat INFERENCE_FRAMEWORK::image_pro(cv::Mat image) {
    cv::Mat inputBlob;
    if (pixel_norm == "(0,1)") {
        inputBlob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(wid, hig), cv::Scalar(), false, false);
    } else if (pixel_norm == "(-1,1)") {
        inputBlob = cv::dnn::blobFromImage(image, 1.0, cv::Size(wid, hig), cv::Scalar(), false, false);
        inputBlob -= 128;
        inputBlob = inputBlob / 128.0;
    } else {
        inputBlob = cv::dnn::blobFromImage(image, 1.0, cv::Size(wid, hig), cv::Scalar(), false, false);
        inputBlob -= 128;
        inputBlob = inputBlob / 128.0;
    }
    return inputBlob;
}

cv::Mat INFERENCE_FRAMEWORK::judge(cv::Mat &inputBlob) {
    net.setInput(inputBlob, net_input_name);
    auto layer = net.getLayer(0);
    cv::Mat forward_result = net.forward(net_output_name);
    return forward_result;
}

cv::Mat INFERENCE_FRAMEWORK::infer(cv::Mat image) {
    cv::Mat inputBlob = image_pro(image);
    cv::Mat out = judge(inputBlob);
    return out;
}