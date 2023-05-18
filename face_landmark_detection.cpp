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
    try
    {
        // Load the pre-trained face detection and landmark detection models
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

        // Load the input image
        cv::Mat inputImage = cv::imread("test_img/1.jpg");

        // Convert the image from OpenCV format to dlib's image format
        cv_image<bgr_pixel> dlibImage(inputImage);

        // Detect faces in the image
        std::vector<rectangle> faceRectangles = detector(dlibImage);

        // Iterate over each detected face
        for (const rectangle& faceRect : faceRectangles)
        {
            // Find the face landmarks
            full_object_detection landmarks = sp(dlibImage, faceRect);

            // Draw the landmarks on the image
            for (unsigned int i = 0; i < landmarks.num_parts(); i++)
            {
                const point& landmark = landmarks.part(i);
                cv::circle(inputImage, cv::Point(landmark.x(), landmark.y()), 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        // Display the result
        cv::imshow("Face Landmark Detection", inputImage);
        cv::waitKey(0);
    }
    catch (const std::exception& e)
    {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}

