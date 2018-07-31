#include "crack.hpp"

Crack c1;
int DEBUG,cnn_mode;

int main(int argc, char *argv[]){

			vector<vector<Point> > contour;
			Mat src,bin;

			if(argc<4)
			{
					cout<<"Usage : "<<endl<<endl;
					cout<<"./Crack <image_path> cnn_mode[ 0 | 1 | 2 ] debug_mode[ 0 | 1 | 2]"<<endl<<endl;
					cout<<"cnn Mode : \n0 - No cnn\n1 - cnn Predict Mode\n"\
				 	<<"2 - cnn Train Database creation mode\n"\
					<<"\nFor Debug mode : \n0 - To disable debug messages and results of intermediate operations\n"\
					<<"1 - To enable debug messages and results of intermediate operations\n"\
					<<"2 - To enable Extended debug messages\n\n"\
					<<"Example : ./Crack 1.jpg 1 0, to run with cnn predict mode and debug off" <<endl;
					return EXIT_FAILURE;
			}

			std::string path,filename,filename_noext;
			path=argv[1];
			filename = path.substr(path.find_last_of("/")+1);
			filename_noext=filename.substr(0,(filename.find_last_of(".")));

			switch ((char)argv[2][0]) {
				case '0':
				cnn_mode=0;
				break;
				case '1':
				cnn_mode=1;
				break;
				case '2':
				cnn_mode=2;
				break;
				default: cout << "Wrong input !!!!";
				exit(1);
				break;
			}

			if(argv[3][0]=='0')
					DEBUG=0;
			else if(argv[3][0]=='1')
					DEBUG=1;
			else if(argv[3][0]=='2')
					DEBUG=2;
			else{
					cout<<"\nWrong value of debug_mode"<<endl;
					exit(1);
			}

			src= imread(argv[1], CV_LOAD_IMAGE_COLOR);		//Reading source image
			if( !src.data ){
						cout<<"Unable to read the input image"<<endl;
	        	return EXIT_FAILURE;
	    }
			resize(src,c1.srcImage,Size(640,480));			//Resizing the image to a constant size

			::google::InitGoogleLogging(argv[0]);

			cvtColor( c1.srcImage, c1.grayImage, COLOR_BGR2GRAY );	//Converting the color image to grayscale

			//Detecting edges using autocanny edge detector
			bin = c1.auto_Canny(c1.grayImage);
#if 0
			if(DEBUG==1 || DEBUG==2){
                    imshow( "Source", c1.srcImage );
					waitKey();
					imshow("AUTO Canny", bin);
					waitKey();
			}
#endif
			contour=c1.Detect_Contour(bin,0);			//Detecting contours after removal of straight lines
			contour=c1.Coordinate(contour,10);			//Removing small unwanted noise from the image

			if(cnn_mode==0){
						c1.finalCracks.clear();
						c1.finalCracks.resize(contour.size());
						c1.finalCracks=contour;
			}

			if(cnn_mode==1){
						c1.finalCracks.clear();
						c1.NonCrack.clear();
                        //imshow( "Source", c1.srcImage );
						std::cout << "CNN Prediction" << '\n';
						std::cout << "---------- Prediction for "
					            << filename << " ----------" << std::endl;
						c1.cnn_predict(c1.srcImage,bin,contour,filename_noext);
			}

			if(cnn_mode==2){
						c1.finalCracks.clear();
						c1.NonCrack.clear();
						imshow("SOURCE",c1.srcImage);
						printf("Enter 1 to Segment and add this image to training database or 0 to skip\n");
						int no = cv::waitKey();
						std::cout << no << '\n';
						if(49==no){
							 c1.Segment(c1.srcImage,bin,contour,filename_noext);
						}
			 			else
						 		cout<<"Skipping...!!!"<<endl;
			}

			Mat final=c1.srcImage.clone();
			for (int i = 0; i < c1.finalCracks.size(); i++){
				cv::Rect boundingRect = cv::boundingRect(c1.finalCracks[i]);
				if(c1.isStraight(c1.finalCracks[i])){
					drawContours( final, c1.finalCracks, i, Scalar(255,0,0),1, 1, 0, 0, Point() );
					cv::rectangle(final, boundingRect, Scalar(0,255,0), 1.5);
		   		if(DEBUG==2){
	      		cout<<(i+1)<<": "<<"CONTOUR SIZE "<<contourArea(c1.finalCracks[i]);
						cout<<" CRACK:"<<(i+1)<<" X:"<<boundingRect.x<<" Y:"<<boundingRect.y<<" W:"<<boundingRect.width<<" H:"<<boundingRect.height<<endl;
					}
				}
			}

			if(cnn_mode==1){
							imwrite(filename,  final );
	//						moveWindow("Cnn Prediction", 750,0);
			}
			else if(cnn_mode==2){
							imwrite(filename ,  final );
							//moveWindow("Cnn Database creation", 750,0);
			}
			else if(cnn_mode==0){
							imwrite(filename ,  final );
							//moveWindow("Cnn Disabled", 750,0);
			}
				cout<<"Total Cracks :"<<c1.finalCracks.size()<<endl;
//			waitKey();
}
