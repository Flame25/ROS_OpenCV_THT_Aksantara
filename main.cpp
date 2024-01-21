#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char **argv){
  //Parser for Input File
  cv::CommandLineParser parser(argc,argv,"{@input||input video}");
  cv::VideoCapture cap(cv::samples::findFile(parser.get<cv::String>("@input")));
  if(!cap.isOpened())
  {
    std::cerr<<"Could not open or find the video\n"<<std::endl; 
  }

  int thresh = 185;
  
  // Opening Video
  while(cap.isOpened()){
    cv::Mat frame; 
    cap >> frame;

    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    if(frame.empty()){
      break; 
    }

    const int max_thresh = 255;
    cv::Mat canny_output;
    cv::Canny(frame_gray, canny_output, thresh, thresh * 2, 3);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_output, contours, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
      mu[i] = moments(contours[i]);
    }
    std::vector<cv::Point2f> mc(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
      // add 1e-5 to avoid division by zero
      mc[i] = cv::Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
                          static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
      std::cout << "mc[" << i << "]=" << mc[i] << std::endl;
    }
    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
   
    for (size_t i = 0; i < contours.size(); i++) {
      cv::Scalar color = cv::Scalar(255,0,0);
      cv::circle(frame, mc[i], 4, color, -1);
      cv::Rect rect = cv::boundingRect(contours[0]);
      cv::rectangle(frame,rect,color,2);
    }
    //cv::imshow("Contours", drawing);
    cv::imshow("Frame", frame);
    std::cout << "\t Info: Area and Contour Length \n";
    for (size_t i = 0; i < contours.size(); i++) {
      std::cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed
                << std::setprecision(2) << mu[i].m00
                << " - Area OpenCV: " << contourArea(contours[i]) << std::endl; 
    }
    if (cv::waitKey(1) == 27) break;
  }
  cap.release();
  cv::destroyAllWindows();
  return 0;
}

