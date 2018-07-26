#include "crack.hpp"
string model_file   = "../model/deploy_crack.prototxt";
string trained_file = "../model/caffe_alexnet_train_iter_450000.caffemodel";
string mean_file    = "../model/crack_mean.binaryproto";
string label_file   = "../model/labels.txt";

/**
  * Detect the Contour in the binary Image
  *@param  bin - Input binary image
  *@return     - vector containing the contour
  *
  *
  */

vector<vector<Point> > Crack:: Detect_Contour(Mat bin, int area,int canny)
{
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
      findContours( bin, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			vector<vector<Point> >cracks( contours.size() );
			cracks.clear();
			for( int i = 0; i< contours.size(); i++){
          if(canny==1){
              if(hierarchy[i][2] < 0 && hierarchy[i][3] < 0) continue;
					 		if (cv::contourArea(contours[i]) > area)
				 			{
				 					cracks.push_back(contours[i]);
				 			}
				 	}
					else{
							cracks.push_back(contours[i]);
					}
			}
			return cracks;
}

/// Find Euclidean Distance between points
double Crack::Distance_BtwnPoints(Point p, Point q)
{
	    int X_Diff = p.x - q.x;
	    int Y_Diff = p.y - q.y;

	    return sqrt((X_Diff * X_Diff) + (Y_Diff * Y_Diff));
}

/**
  *checking particular char is y,Y,n,N
  *@param  ch - character const
  *@return true/false
  *
  */
bool check_char( char ch)
{
			if(ch=='Y' || ch=='y' || ch=='N' ||ch=='n')
						return true;
			return false;
}

/**
  * Converting the contours to polylines using approxPolyDP function
  *@param  bin       - Input image
  *@return     - Mat image with contours in polylines
  *
  */
Mat Crack::Approx_Poly(Mat bin)
{
			vector<vector<Point> > contours,cracks;
			Mat drawing = Mat::zeros( bin.size(),  CV_8UC1 );
			cracks=Detect_Contour(bin,0,0);
		  contours.resize(cracks.size());
		  for( size_t k = 0; k < cracks.size(); k++ )
		   				approxPolyDP(Mat(cracks[k]), contours[k], 2, true);

			Scalar color = Scalar(255,255,255 );
		  for(int i=0;i<contours.size();i++)
		  				drawContours( drawing,contours , i, color,1, 0, noArray(), 0, Point() );
    	return drawing;
}

/**
  * Find distance between xmin,ymin and xmax,ymax co-ordinates and decide if noise or not
  *@param  contour  - vector of Points
  *@return bool - return if Noise or not
  *
*/

bool Crack::Coordinate_difference(vector<Point> contour,int tdiff)
{
			int xmin,xmax,ymin,ymax,xdiff,ydiff,diff;
			xmax=ymax=0;
			xmin=contour[0].x;
			ymin=contour[0].y;
		  for(unsigned int j=0;j<contour.size();j++)
		  {
		     		if(xmin >= contour[j].x)
		     		{
		     			xmin = contour[j].x;
		     		}

		     		if(ymin >= contour[j].y)
		     		{
		     			ymin = contour[j].y;
		     		}

		     		if(xmax < contour[j].x)
		     		{
		     			xmax = contour[j].x;
		     		}

		     		if(ymax < contour[j].y)
		     		{
		     			ymax = contour[j].y;
		     		}
			}
		  xdiff=xmax-xmin;
			ydiff=ymax-ymin;
			diff=xdiff-ydiff;

			if(diff<0)
			{
					diff= -diff;
			}

    	if(diff > tdiff && ( xdiff>25 ) || ( ydiff>25 ) )
			{
					if(DEBUG==2)
						cout<<"True diff="<<diff<<endl;
					return true;
			}else
			{
					if(DEBUG==2)
						cout<<"False diff="<<diff<<endl;
					return false;
			}
}

/**
  * Removing unwanted noise by calculating the x and y coordinate area of the contour
  *@param  contours  - vector containing the contours
  *@param  tdiff     - minimum difference required
  *@return crack     - contours after removing unwanted noise
  *
*/
vector<vector<Point> > Crack::Coordinate( vector<vector<Point> > contours ,int tdiff )
{
			bool diff;
			vector<vector<Point> >crack;
			int j=0;
			for(unsigned int i=0;i<contours.size();i++)
			{
			     diff=Coordinate_difference(contours[i],tdiff);
						if(diff==true){
									crack.push_back(contours[i]);
            }
			}
			return crack;
}

/**
  * Checking if a given contour is a straight line or not
  *@param  contours  - vector containing the contours
  *@return bool - return if straight line or not
  *
*/

bool Crack::isStraight(std::vector<Point> contours){
	cv::Rect boundingRect = cv::boundingRect(contours);
  double diagonal_length=Distance_BtwnPoints(Point(boundingRect.x,boundingRect.y),Point(boundingRect.x+boundingRect.width,boundingRect.y+boundingRect.height));
  double ratio=diagonal_length/contours.size();
	double ratio_threshold=2.02;
	if(DEBUG==2)
    std::cout << "diagonal_length : "<<diagonal_length << "Ratio is : "<< ratio << '\n';
  if(ratio<ratio_threshold)
		return true;//Is not a straight line
  else
    return false;//Is a straight line
}

/**
  * Find Median value of the Mat Image
  *@param  Input - Input grayscale image
  *@return  - Median of the Mat
  *
  */

double Crack::medianMat(cv::Mat Input) {

			Input = Input.reshape(0, 1); // spread Input Mat to single row
			std::vector<double> vecFromMat;
			Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
			std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
			return vecFromMat[vecFromMat.size() / 2];
}

/**
  *
  *@param  src_gray - Input grayscale image
  *@param  lowThreshold - Threshold for canny edge detector
  *@return  - image containing the valid contours detected by Canny edge detector
  *
  */

Mat Crack::auto_Canny(Mat src_gray, double sigma) {
			// compute the median of the single channel pixel intensities
			double v = medianMat(src_gray);
			Mat edged,temp;
      vector<vector<Point> > contour;

			// apply automatic Canny edge detection using the computed median
			int lower = int(std::max(0.0, (1.0 - sigma) * v));
			int upper = int(std::min(255.0, (1.0 + sigma) * v));

      blur(src_gray, temp, Size(3, 3));
			Canny(temp, temp, lower, upper);
      contour=Detect_Contour(temp,0,0);
      if(contour.size()>=60){
        std::cout << "contour size : " << contour.size()<< '\n';
        blur(src_gray, src_gray, Size(3, 3));
      }
      contour.clear();

			int area = 5;
			int i=1;
			do{
			blur(src_gray, edged, Size(3, 3));

			/// Canny detector
			Canny(edged, edged, lower, upper);

			//Detecting contour
			contour=Detect_Contour(edged,area,1);
			area=area+10;

			v = medianMat(edged);
			lower = lower + int(std::max(0.0, (1.0 - sigma) * v));
			upper = int(std::min(255.0, (1.0 + sigma) * v));
    }while(contour.size()>=40);
			return edged;
}

/**
  * Removing contours that lie completely in another contour
  *@param  contours  - vector containing the contours
  *@return vector containing valid contours
  *
  */

vector<vector<Point> > Crack::remove_duplicates(vector<vector<Point> > contours){
	int flag=0;
	vector< Point > temp;
	vector<vector<Point> > F_Cracks;

	for (int i = 0; i < contours.size(); i++){
		Rect r_sample=cv::boundingRect(contours[i]);
		temp=contours[i];
		flag=0;
		for(int r=0; r<contours.size();r++){
			if(i==r) {
				continue;
			}
			Rect r_test=cv::boundingRect(contours[r]);
			if((r_sample & r_test) == r_sample){
				flag=1;
				break;
			}
		}
		if(flag==0){
			F_Cracks.push_back(temp);
		}
	}
	return F_Cracks;
}

/**
  * Segment the contours as 20*10 images for cnn Database
  *@param  src  - Source Mat Image
	*@param  binobj  - Binary Image
	*@param  pcontour  - vector containing the contours
	*@param  Fname  - Name of the current Image
  *@return void
  *
  */

void Crack::Segment(Mat src,Mat binObj,vector<vector<Point> > pcontour,string Fname)
{
	Mat src1=src.clone();
	int box_w=10; // Define box width here
	int box_h=10; // Define box height here
	int threshold_perc=1; //perceantage value for eliminating the box according to pixel count inside the box
	int threshold=(box_w*box_h*threshold_perc)/100;
	vector< Vec4i > hierarchy;
	Mat dest1=src.clone();
	int l=0;
	int flag=0;
	vector<vector<Point> > F_Cracks;
	stringstream ss;
	string path = "./Contours/";
	string name = "_cropped_";
	string type = ".jpg";
	Mat matROI,matROI_src;
	string filename;
	ofstream myfile;
  myfile.open ("./Contours/labels.txt",ios::app);

	F_Cracks = remove_duplicates(pcontour);

	for (int i = 0; i < F_Cracks.size(); i++)
	{
		 cv::Rect boundingRect = cv::boundingRect(F_Cracks[i]);
		 cv::rectangle(src, boundingRect, cv::Scalar(0, 255, 0), 2);      // draw red rectangle around each contour as we ask user for input
		 drawContours(src, F_Cracks, i, cv::Scalar(255, 0, 0), 1, 0, 0, 0, Point());
		 cv::imshow("imgTrainingCrack", src);
	   LABEL:  cout<<"Enter Y for crack or N for Uncrack"<<endl;
		 int intChar = cv::waitKey(0);
		 int w_delta=5;
		 int h_delta=5;
		 int rotate_img=0;
		 if (intChar == 27)
		 {
			 cout<<"Terminating the program"<<endl;
			 exit(1);
		 }else if(check_char((char)intChar)){
				 // Scan the image with in bounding box
				 if(boundingRect.width>=boundingRect.height){
					 box_w=20;
					 box_h=10;
				 }
				 else{
					 box_w=10;
					 box_h=20;
					 rotate_img=1;
				 }

				 if(boundingRect.width<box_w){
					 	box_w=boundingRect.width;
						if(box_w<w_delta)
							w_delta=box_w;
				 }

				 if(boundingRect.height<box_h){
					 	box_h=boundingRect.height;
						if(box_h<h_delta)
							h_delta=box_h;
				 }

				 threshold=(box_w*box_h*threshold_perc)/100;
				 	int W_MAX=(boundingRect.x+boundingRect.width);
					int H_MAX=(boundingRect.y+boundingRect.height);

						for(int j=boundingRect.x; (j+box_w)<=W_MAX; j=j+w_delta){
							if(w_delta==0)break;
							w_delta=((W_MAX-(j+box_w))<w_delta)?(W_MAX-(j+box_w)):w_delta;
							h_delta=5;
							if(box_h<h_delta) h_delta=box_h;
							for(int k=boundingRect.y;((k+box_h)<=H_MAX); k=k+h_delta){
								if(h_delta==0)break;
								h_delta=((H_MAX-(k+box_h))<h_delta)?(H_MAX-(k+box_h)):h_delta;
								Rect roi_rect(j,k,box_w,box_h);
								matROI = binObj(roi_rect);
								matROI_src = src1(roi_rect);
								int count = countNonZero(matROI);
								if((count > threshold) &&(count > 0)){
									if((char)intChar=='Y' || (char)intChar=='y'){
										rectangle(dest1, roi_rect, Scalar(0,255,0),1,8,0 );
										ss<<"crack_"<<Fname<<name<<l<<type;
										myfile << ss.str() <<"\t"<<0<<"\n";
									}
									else{
										rectangle(dest1, roi_rect, Scalar(255,0,255),1,8,0 );
										ss<<"noncrack_"<<Fname<<name<<l<<type;
										myfile << ss.str() <<"\t"<<1<<"\n";
									}
									cv::imshow("Crack Contours", dest1);
									++l;
									filename = ss.str();
									if(rotate_img== 1){
										flip(matROI_src.t(),matROI_src,1);
										if(DEBUG==2)
											std::cout << "Image rotated" << '\n';
									}

									if(matROI_src.size().width<20 || matROI_src.size().height<10){
										resize(matROI_src,matROI_src,Size(20,10));
										if(DEBUG==2)
											std::cout << "Image resized to 20 x 10" << '\n';
									}

									imwrite(path+filename, matROI_src);
									ss.str("");
								}
								else{
									rectangle(dest1, roi_rect, Scalar(0,0,255),1,8,0 );
									cv::imshow("Crack Contours", dest1);
								}
							}
						}
			}
			else if(115 == intChar || 83 == intChar)
			{
				cout<<"Skiping"<<endl;
			}
			else
			{
				cout<<"Please Enter correct value"<<endl;
				goto LABEL;
			}
	}
	cv::imshow("Crack Contours", dest1);
	waitKey();
	destroyWindow("imgTrainingCrack");
	myfile.close();
}

/**
  * Predict whether the contours are cracks or not
  *@param  src  - Source Mat Image
	*@param  binobj  - Binary Image
	*@param  pcontour  - vector containing the contours
	*@param  Fname  - Name of the current Image
  *@return void
  *
  */

void Crack::cnn_predict(Mat src,Mat binObj,vector<vector<Point> > pcontour,string Fname)
{
	Classifier classifier(model_file, trained_file, mean_file, label_file);
	Mat src1=src.clone();
	int box_w=20; // Define box width here
	int box_h=10; // Define box height here
	int threshold_perc=1; //perceantage value for eliminating the box according to pixel count inside the box
	int threshold=(box_w*box_h*threshold_perc)/100;
	vector< Vec4i > hierarchy;
	Mat dest1=src.clone();
	int i=0;
	int l=0;
	int flag=0;
	vector<vector<Point> > F_Cracks;
	Mat matROI,matROI_src;

	int crack_threshold_perc=70; //perceantage value for deciding whether the whole contour is a Crack or Not
	int crack_threshold=0;
	int crack_count=0;
	int t_count=0;
	int c_count=0;

	F_Cracks = remove_duplicates(pcontour);

	finalCracks.erase(finalCracks.begin(),finalCracks.end());
	NonCrack.erase(NonCrack.begin(),NonCrack.end());

	for (i = 0; i < F_Cracks.size(); i++)
	{
		 cv::Rect boundingRect = cv::boundingRect(F_Cracks[i]);
		 int w_delta=5;
		 int h_delta=5;
		 int rotate_img=0;
		 // Scan the image with in bounding box
		 if(boundingRect.width>=boundingRect.height){
			 box_w=20;
			 box_h=10;
		 }
		 else{
					 box_w=10;
					 box_h=20;
					 rotate_img=1;
			}

			if(boundingRect.width<box_w){
				 	box_w=boundingRect.width;
					if(box_w<w_delta)
						w_delta=box_w;
			}

			if(boundingRect.height<box_h){
			 	box_h=boundingRect.height;
				if(box_h<h_delta)
					h_delta=box_h;
			}

			threshold=(box_w*box_h*threshold_perc)/100;
			int W_MAX=(boundingRect.x+boundingRect.width);
			int H_MAX=(boundingRect.y+boundingRect.height);

			t_count=0;
			crack_count=0;
			for(int j=boundingRect.x; (j+box_w)<=W_MAX; j=j+w_delta){
				if(w_delta==0)break;
				w_delta=((W_MAX-(j+box_w))<w_delta)?(W_MAX-(j+box_w)):w_delta;
				h_delta=5;
				if(box_h<h_delta) h_delta=box_h;
				for(int k=boundingRect.y;((k+box_h)<=H_MAX); k=k+h_delta){
					if(h_delta==0)break;
					h_delta=((H_MAX-(k+box_h))<h_delta)?(H_MAX-(k+box_h)):h_delta;
					Rect roi_rect(j,k,box_w,box_h);
					matROI = binObj(roi_rect);
					matROI_src = src1(roi_rect);
					int count = countNonZero(matROI);
					if((count > threshold) &&(count > 0)){
						t_count++;
						if(rotate_img== 1){
							flip(matROI_src.t(),matROI_src,1);
							if(DEBUG==2)
								std::cout << "Image rotated" << '\n';
						}
						if(matROI_src.size().width<20 || matROI_src.size().height<10){
							resize(matROI_src,matROI_src,Size(20,10));
							if(DEBUG==2)
								std::cout << "Image resized to 20 x 10" << '\n';
						}

						std::vector<Prediction> predictions = classifier.Classify(matROI_src);

						/* Print the top N predictions. */
						for (size_t i = 0; i < predictions.size(); ++i) {
							Prediction p = predictions[i];
							String Crack_str="Crack";
							if(strcmp(p.first.c_str(),Crack_str.c_str())==0){
								if((p.second*100)>=50){
									crack_count++;
									cv::rectangle(dest1, roi_rect, Scalar(0,255,0), 1.5);
								}
								else{
									cv::rectangle(dest1, roi_rect, Scalar(0,0,255), 1.5);
								}
							}
						}
					}
					else{
							continue;
					}
				}
			}
			crack_threshold=(t_count*crack_threshold_perc)/100;
			double perc=0;
			perc=((double)crack_count/(double)t_count)*100;
			if(DEBUG==1 || DEBUG ==2){
				cout<<"PASSED_CRACK_COUNT "<<crack_count<<" crack_threshold "<<crack_threshold<<" total_crack "<<t_count<<endl;
				std::cout << "percentage: " << perc<< '\n';
			}

			if(crack_count>=crack_threshold){
					if(DEBUG==1 || DEBUG ==2)
						cout<<"Crack"<<endl;
					finalCracks.push_back(F_Cracks[i]);
					cv::Rect t_boundingRect = cv::boundingRect(F_Cracks[i]);
					drawContours( dest1, F_Cracks, i, Scalar(255,0,0),1, 1, 0, 0, Point() );
					cv::rectangle(dest1, t_boundingRect, Scalar(0,255,0), 1.5);
					c_count++;
			}
			else{
				if(DEBUG==1 || DEBUG ==2)
					cout<<"NON Crack"<<endl;
				NonCrack.push_back(F_Cracks[i]);
				cv::Rect t_boundingRect = cv::boundingRect(F_Cracks[i]);
				drawContours( dest1, F_Cracks, i, Scalar(255,0,0),1, 1, 0, 0, Point() );
				cv::rectangle(dest1, t_boundingRect, Scalar(0,0,255), 1.5);
				c_count++;
			}
			if(DEBUG==1 || DEBUG ==2){
				cv::imshow("Crack Contours", dest1);
				waitKey(1);
			}
	}
	if(DEBUG==1 || DEBUG ==2)
		std::cout << "Total Contours: "<<c_count << '\n';
	c_count=0;
}
