#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

bool defect_exists(const cv::Mat& input, const cv::Mat& reference, cv::Mat& marked_image);

int main(int argc, char** argv)
{
    using std::cout;
    using std::string;
    namespace po = boost::program_options;
    namespace fs = boost::filesystem;

    auto desc = po::options_description("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("input,i", po::value<string>(), "input to be tested")
        ("reference,r", po::value<string>(), "reference image");

    auto args_map = po::variables_map();
    po::store(po::parse_command_line(argc, argv, desc), args_map);
    po::notify(args_map);

    if (args_map.count("help")
        || !args_map.count("input")
        || !args_map.count("reference"))
    {
        cout << desc << "\n";
        return 1;
    }

    if (!fs::exists(fs::path(args_map["reference"].as<string>())))
    {
        cout << "File " << args_map["reference"].as<string>() << " does not exist!\n";
        return 1;
    }

    if (!fs::exists(fs::path(args_map["input"].as<string>())))
    {
        cout << "File " << args_map["input"].as<string>() << " does not exist!\n";
        return 1;
    }

    auto reference_image = cv::imread(args_map["reference"].as<string>());
    auto input_image = cv::imread(args_map["input"].as<string>());
    auto marked_image = cv::Mat();

    if (defect_exists(input_image, reference_image, marked_image))
    {
        cout << "Defect found\n";
        cv::imshow("Defect found!", marked_image);
        cv::waitKey(0);
        cv::imwrite("defect.jpg", marked_image);
    }
    else
    {
        cout << "No defects found\n";
    }

    return 0;
}

bool defect_exists(const cv::Mat& input, const cv::Mat& reference, cv::Mat& marked_image)
{
    cv::Mat gray_input;
    cv::Mat gray_reference;
    //Convert to grayscale -> histeq -> thresholding -> closing

    cv::cvtColor(input, gray_input, CV_RGB2GRAY);
    cv::cvtColor(reference, gray_reference, CV_RGB2GRAY);

    cv::equalizeHist(gray_input, gray_input);
    cv::equalizeHist(gray_reference, gray_reference);

    auto threshold = 150;
    cv::threshold(gray_input, gray_input, threshold, 255, cv::THRESH_BINARY);
    cv::threshold(gray_reference, gray_reference, threshold, 255, cv::THRESH_BINARY);

    auto strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(gray_input, gray_input, cv::MORPH_CLOSE, strel);
    cv::morphologyEx(gray_reference, gray_reference, cv::MORPH_CLOSE, strel);

    //XOR to detect differences
    cv::Mat xor_result;
    cv::bitwise_xor(gray_input, gray_reference, xor_result);
    //Get rid of noise
    cv::medianBlur(xor_result, xor_result, 3);

    //If there are non-zero pixels
    if (cv::countNonZero(xor_result) > 1)
    {
        //Outline the defects in red.
        auto output = input.clone();

        //First color the defect outlines black.
        cv::Mat defect_mask = 255 - xor_result;
        cv::cvtColor(defect_mask, defect_mask, CV_GRAY2RGB);
        cv::bitwise_and(output, defect_mask, output);

        //Change the white pixels in the outline to red.
        auto color_xor = cv::Mat();
        auto dilation_strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(xor_result, color_xor, dilation_strel, cv::Point(-1, -1));
        cv::cvtColor(color_xor, color_xor, CV_GRAY2RGB);
        for (auto pixel = color_xor.begin<cv::Vec3b>(); pixel != color_xor.end<cv::Vec3b>(); pixel++)
        {
            if ((*pixel)[0] > 0
                || (*pixel)[1] > 0
                || (*pixel)[2] > 0)
            {
                *pixel = cv::Vec3b(0, 0, 255);
            }
        }

        //Blend the masked output and the red outline.
        cv::addWeighted(output, 0.4, color_xor, 1.0, 0, output);

        marked_image = output;
        return true;
    }
    else
    {
        return false;
    }
}
