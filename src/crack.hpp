#include<iostream>
#include <fstream>
#include<string.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include "predict.hpp"

using namespace std;
using namespace cv;

extern int DEBUG,cnn_mode;

extern string model_file;//   = "/home/nj/work/Crack_CNN/integrated_code/New_model/deploy_crack.prototxt";
extern string trained_file;// = "/home/nj/work/Crack_CNN/integrated_code/New_model/caffe_alexnet_train_iter_50000.caffemodel";
extern string mean_file;//    = "/home/nj/work/Crack_CNN/integrated_code/New_model/crack_mean.binaryproto";
extern string label_file;//   = "/home/nj/work/Crack_CNN/integrated_code/New_model/labels.txt";

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Crack {
	public:
		Mat srcImage,grayImage;
		vector<vector<Point> > finalCracks,NonCrack;

		Mat Approx_Poly(Mat bin);
		Mat auto_Canny(Mat src_gray, double sigma = 0.33);
		vector<vector<Point> > Detect_Contour(Mat bin, int area,int canny=0);
		vector<vector<Point> > remove_duplicates(vector<vector<Point> > contours);
		vector<vector<Point> > Coordinate( vector<vector<Point> > contours ,int tdiff );
		double medianMat(cv::Mat Input);
		double Distance_BtwnPoints(Point p, Point q);
		bool isStraight(std::vector<Point> contours);
		bool Coordinate_difference(vector<Point> a,int tdiff);
		void Segment(Mat src,Mat binObj,vector<vector<Point> > pcontour,string Fname);
		void cnn_predict(Mat src,Mat binObj,vector<vector<Point> > pcontour,string Fname);
};
