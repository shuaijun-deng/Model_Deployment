//
// Created by dell on 20-7-13.
//

#pragma once

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


class INFERENCE_FRAMEWORK {
public:
    void init(std::string model_file, int width, int heigth, std::string pixel_normalization, std::string net_input,
              std::string net_output);

    cv::Mat infer(cv::Mat image);

private:
    int wid;
    int hig;
    std::string model;
    std::string pixel_norm;
    std::string net_input_name;
    std::string net_output_name;

    cv::dnn::Net net;

    cv::Mat image_pro(cv::Mat image);

    cv::Mat judge(cv::Mat &inputBlob);
};