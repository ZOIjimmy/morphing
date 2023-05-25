#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/deps/file_util.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/calculators/core/concatenate_vector_calculator.pb.h"
#include "mediapipe/calculators/core/concatenate_vector_calculator.h"
#include "mediapipe/calculators/tflite/tflite_inference_calculator.pb.h"
#include "mediapipe/calculators/tflite/tflite_inference_calculator.h"
#include "mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.h"
#include "mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

void ExtractFacialLandmarks(const cv::Mat& image) {
  // Create a MediaPipe graph
  std::string graph_config =
      "pbtxt_path: \"path/to/your/mediapipe/graph.pbtxt\"\n"
      "node {\n"
      "  calculator: \"ConcatenateVectorCalculator\"\n"
      "  input_stream: \"IMAGE:input_video\"\n"
      "  input_stream: \"FRAMES:input_frames\"\n"
      "  output_stream: \"CONCATENATED_VECTOR:concatenated_vector\"\n"
      "}\n"
      "node {\n"
      "  calculator: \"TfLiteInferenceCalculator\"\n"
      "  input_stream: \"CONCATENATED_VECTOR:concatenated_vector\"\n"
      "  output_stream: \"TENSORS:tensors\"\n"
      "}\n"
      "node {\n"
      "  calculator: \"TfLiteTensorsToDetectionsCalculator\"\n"
      "  input_stream: \"TENSORS:tensors\"\n"
      "  output_stream: \"DETECTIONS:detections\"\n"
      "}\n"
      "node {\n"
      "  calculator: \"TfLiteTensorsToLandmarksCalculator\"\n"
      "  input_stream: \"TENSORS:tensors\"\n"
      "  output_stream: \"LANDMARKS:facial_landmarks\"\n"
      "}";
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_config);

  // Initialize the graph
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  // Start running the graph
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  // Create an OpenCV packet for the input image
  mediapipe::Packet image_packet =
      mediapipe::Adopt(new mediapipe::ImageFrame(
          mediapipe::ImageFormat::SRGB, image.cols, image.rows,
          mediapipe::ImageFrame::kDefaultAlignmentBoundary));

  // Input image to the graph
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "input_video", mediapipe::Adopt(image_packet).At(mediapipe::Timestamp(0))));

  // Get the output packets from the graph
  mediapipe::Packet landmarks_packet;
  MP_RETURN_IF_ERROR(graph.GetOutputPacket("facial_landmarks", &landmarks_packet));

  // Convert the landmarks packet to a LandmarkList
  const auto& landmarks = landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();

  // Process the landmarks as needed
  for (const auto& landmark : landmarks.landmark()) {
    // Access individual landmark points
    float x = landmark.x();
    float y = landmark.y();
    float z = landmark.z();

    // Process the landmarks as needed
    // ...
  }

  // Close the graph
  MP_RETURN_IF_ERROR(graph.CloseInputStream("input_video"));

  // Wait until the graph finishes processing
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
}
int main() {
  cv::Mat image = cv::imread("test_img/1.jpg");
  if (image.empty()) {
    std::cerr << "Failed to load image." << std::endl;
    return 1;
  }
  ExtractFacialLandmarks(image);
  return 0;
}
