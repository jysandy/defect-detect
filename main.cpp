#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

bool defect_exists(const cv::Mat& input, const cv::Mat& reference);
cv::Mat edge_image(cv::Mat in);

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

    if (defect_exists(input_image, reference_image))
    {
        cout << "Defect found\n";
    }
    else
    {
        cout << "No defects found\n";
    }

    return 0;
}

bool defect_exists(const cv::Mat& input, const cv::Mat& reference)
{
    cv::Mat gray_input;
    cv::Mat gray_reference;
    //Convert to grayscale -> median filter -> histeq -> adaptive thresholding -> opening

    cv::cvtColor(input, gray_input, CV_RGB2GRAY);
    cv::cvtColor(reference, gray_reference, CV_RGB2GRAY);

    cv::medianBlur(gray_input, gray_input, 3);
    cv::medianBlur(gray_reference, gray_reference, 3);

    cv::equalizeHist(gray_input, gray_input);
    cv::equalizeHist(gray_reference, gray_reference);

    cv::adaptiveThreshold(gray_input, gray_input, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY, 3, 0);
    cv::adaptiveThreshold(gray_reference, gray_reference, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY, 3, 0);

    auto strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(gray_input, gray_input, cv::MORPH_OPEN, strel);
    cv::morphologyEx(gray_reference, gray_reference, cv::MORPH_OPEN, strel);

    cv::Mat xor_result;
    cv::bitwise_xor(gray_input, gray_reference, xor_result);
    //Open again to get rid of random noise
    cv::morphologyEx(xor_result, xor_result, cv::MORPH_OPEN, strel);

    //If there are any non-zero pixels then a defect exists.
    return cv::countNonZero(xor_result) > 1;
}

cv::Mat edge_image(cv::Mat in)
{
    cv::Mat grad_x;
    cv::Mat grad_y;
    int depth = CV_16S;

    cv::Sobel(in, grad_x, depth, 1, 0);
    cv::Sobel(in, grad_y, depth, 0, 1);
    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);
    cv::Mat ret;
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, ret);

    return ret;
}
