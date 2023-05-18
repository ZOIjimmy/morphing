#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

int latency=100;
double a, b, p = 0.0;
cv::Vec3b lineColor(0, 255, 0);
int width, height, pair_num, frame_num;
std::string image_name[2], newname;
cv::Mat image[2], image_origin[2];
std::vector<cv::Mat> frames;
int zero = 0;
int one = 1;

struct Line {
	cv::Point2f P;
	cv::Point2f Q;
	cv::Point2f M;
	double len;
	double degree;
};
std::vector<struct Line> lines[2];
Line curLine[2];
void on_mouse(int event, int x, int y, int flag, void* param) {
    int v = *static_cast<int*>(param);
    if (event==CV_EVENT_LBUTTONDOWN||event==CV_EVENT_RBUTTONDOWN) {
        curLine[v].P = cv::Point2f(x, y);
    } else if (event==CV_EVENT_LBUTTONUP||event==CV_EVENT_RBUTTONUP) {
        curLine[v].Q = cv::Point2f(x, y);
        curLine[v].M = (curLine[v].P + curLine[v].Q) * 0.5;
        cv::Vec2f tmp = curLine[v].Q - curLine[v].P;
        curLine[v].len = cv::norm(tmp);
        curLine[v].degree = std::remainder(std::atan2(tmp[1], tmp[0]), 2.0 * M_PI);
        cv::line(image[v], curLine[v].P, curLine[v].Q, lineColor, 2, CV_AA, 0);
        cv::imshow(image_name[v], image[v]);
        if (curLine[v].P != curLine[v].Q) {
            lines[v].push_back(curLine[v]);
        }
    } else if (flag==CV_EVENT_FLAG_LBUTTON||flag==CV_EVENT_FLAG_RBUTTON) {
        curLine[v].Q = cv::Point2f(x, y);
        cv::Mat imagecopy = image[v].clone();
        cv::line(imagecopy,cv::Point(curLine[v].P.x, curLine[v].P.y),cv::Point(curLine[v].Q.x, curLine[v].Q.y),lineColor,2,CV_AA,0);
        cv::imshow(image_name[v], imagecopy);
    }
}
void run(int f) {
    std::vector<Line> warpLines;
    double ratio = static_cast<double>(f+1) / (frame_num+1);
    for (int i=0;i<pair_num;i++) {
        Line newLine;
        newLine.M = (1 - ratio) * lines[0][i].M + ratio * lines[1][i].M;
        newLine.len = (1 - ratio) * lines[0][i].len + ratio * lines[1][i].len;
        newLine.degree = (1 - ratio) * lines[0][i].degree + ratio * lines[1][i].degree;
        newLine.P = newLine.M - 0.5 * newLine.len * cv::Point2f(std::cos(newLine.degree), std::sin(newLine.degree));
        newLine.Q = newLine.M + 0.5 * newLine.len * cv::Point2f(std::cos(newLine.degree), std::sin(newLine.degree));
        warpLines.push_back(newLine);
    }
    cv::Mat new_image(cv::Size(width, height), CV_8UC3);
    for (int x=0;x<width;x++) {
        for (int y=0;y<height;y++) {
            cv::Point2f dst_point(x, y);
            cv::Vec3b scalar[2];
            cv::Point2f sum[2] = {cv::Point2f(0,0), cv::Point2f(0,0)};
            for (int j=0;j<pair_num;j++) {
                Line dst_line = warpLines[j];
                cv::Point2f X_P = dst_point - dst_line.P;
                cv::Point2f Q_P = dst_line.Q - dst_line.P;
                cv::Point2f Perp_Q_P(Q_P.y, -Q_P.x);
                double u = X_P.dot(Q_P) / std::pow(dst_line.len, 2);
                double v = X_P.cross(Q_P) / dst_line.len;
                for (int i=0;i<2;i++) {
                    Line src_line = lines[i][j];
                    sum[i] += src_line.P + u * Q_P + (v / src_line.len) * Perp_Q_P;
                }
            }
            for (int i=0;i<2;i++) {
                sum[i].x = std::clamp(sum[i].x / pair_num, 0.0f, static_cast<float>(width-1));
                sum[i].y = std::clamp(sum[i].y / pair_num, 0.0f, static_cast<float>(height-1));
                int x_floor = static_cast<int>(sum[i].x);
                int y_floor = static_cast<int>(sum[i].y);
                int x_ceil = std::min(x_floor + 1, width - 1);
                int y_ceil = std::min(y_floor + 1, height - 1);
                double dx = sum[i].x - x_floor;
                double dy = sum[i].y - y_floor;
                scalar[i] = (1 - dx) * (1 - dy) * image_origin[i].at<cv::Vec3b>(y_floor, x_floor) +
                                dx * (1 - dy) * image_origin[i].at<cv::Vec3b>(y_floor, x_ceil) +
                                dx * dy * image_origin[i].at<cv::Vec3b>(y_ceil, x_ceil) +
                                (1 - dx) * dy * image_origin[i].at<cv::Vec3b>(y_ceil, x_floor);
            }
            new_image.at<cv::Vec3b>(y, x) = (1 - ratio) * scalar[0] + ratio * scalar[1];
        }
    }
    char img_name[20];
    snprintf(img_name,20,"%s_%d.jpg",newname.c_str(),f);
    cv::imwrite(img_name,new_image);
    frames[f+1] = new_image;
}
void showFrames() {
    while (pair_num > 0) {
        for (int i=0;i<frame_num+2;i++) {
            cv::imshow("frames", frames[i]);
            int key = cv::waitKey(latency);
            if (key == 'q') return;
        }
    }
}
void init(int argc, char** argv) {
    image_name[0] = argv[1];
    image_name[1] = argv[2];
    frame_num = argc >= 4 ? std::atoi(argv[3]) : 5;
    a = argc >= 5 ? std::atoi(argv[4]) : 1;
    b = argc >= 6 ? std::atoi(argv[5]) : 2;
    p = argc >= 7 ? std::atoi(argv[6]) : 0;
    newname = argc >= 8 ? argv[7] : "out";

    image_origin[0] = cv::imread(image_name[0].c_str());
    image_origin[1] = cv::imread(image_name[1].c_str());
    image[0] = image_origin[0].clone();
    image[1] = image_origin[1].clone();
  	height = image[0].rows;
  	width = image[0].cols;
    frames.resize(frame_num+2);
    
    cv::namedWindow(image_name[0], cv::WINDOW_NORMAL);
  	cv::moveWindow(image_name[0], 100, 100);
    cv::namedWindow(image_name[1], 1);
    cv::moveWindow(image_name[1], 100+width, 100);
    cv::setMouseCallback(image_name[0], on_mouse, &zero);
    cv::setMouseCallback(image_name[1], on_mouse, &one);
    cv::imshow(image_name[0], image[0]);
    cv::imshow(image_name[1], image[1]);
}
int main(int argc, char** argv){
    if (argc < 3) {
        printf("Give me two images!!!\n");
        return 1;
    }
    init(argc, argv);
    std::thread threads[frame_num];
    frames[0] = image_origin[0];
  	while(1){
		int key = cvWaitKey(0);
		if (key == 13) {
            pair_num = std::min(lines[0].size(),lines[1].size()); 
            for (int i=0;i<frame_num;i++) {
                threads[i] = std::thread(run, i);
            }
            for (int i=0;i<frame_num;i++) {
                threads[i].join();
            }
            break;
        } else if (key == 'q')
			break;
  	}
    frames[frame_num+1] = image_origin[1];
    showFrames();
  	cv::destroyWindow(image_name[0]);
  	cv::destroyWindow(image_name[1]);
	return 0;
}
