#include <opencv2/opencv.hpp>

int main() {
    // Load the image
    cv::Mat image = cv::imread("test_img/1.jpg");

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cout << "Failed to load the image." << std::endl;
        return -1;
    }

    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Detect edges using Canny edge detection
    cv::Mat edges;
    double lowThreshold = 100.0;
    double highThreshold = 100.0;
    cv::Canny(grayImage, edges, lowThreshold, highThreshold);

    // Find contours in the edge image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Minimum length threshold for long edges
    double minLengthThreshold = 100.0;

    // Iterate over the contours and filter long edges
    for (auto it = contours.begin(); it != contours.end(); ++it) {
        double contourLength = cv::arcLength(*it, true);
        if (contourLength > minLengthThreshold) {
            // Draw lines on the original image for long edges
            cv::Scalar color(0, 255, 0); // Green color
            cv::drawContours(image, contours, static_cast<int>(std::distance(contours.begin(), it)), color, 2);
        }
    }


    // Display the original image with the drawn lines
    cv::imshow("Long Edges", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

