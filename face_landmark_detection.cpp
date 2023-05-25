#include <iostream>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

using namespace dlib;

int main()
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    cv::Mat inputImage = cv::imread("test_img/1.jpg");
    cv_image<bgr_pixel> dlibImage(inputImage);
    std::vector<rectangle> faceRectangles = detector(dlibImage);
    for (const rectangle& faceRect : faceRectangles) {
        full_object_detection landmarks = sp(dlibImage, faceRect);
        for (unsigned int i = 0; i < landmarks.num_parts(); i++) {
            const point& landmark = landmarks.part(i);
            cv::circle(inputImage, cv::Point(landmark.x(), landmark.y()), 2, cv::Scalar(0, 255, 0), -1);
        }
    }
    cv::imshow("Face Landmark Detection", inputImage);
    cv::waitKey(0);
    return 0;
}

