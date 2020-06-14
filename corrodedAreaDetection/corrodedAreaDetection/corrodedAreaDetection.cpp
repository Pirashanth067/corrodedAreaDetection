#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <list>

using namespace cv;
using namespace std;

Mat src, imBin,dst,src_cpy,patchMat,finalMat, test;
int MAX_KERNEL_LENGTH = 31;
int thresh = 180;
int max_thresh = 255;
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int s = 0,t=0,totrow,totcol;
char window_name[] = "Corrosion Detector";
double  mythreshold;
int threshold_value,totalArea=0,corrodedArea=0;


void thresh_callback(int, void*); //funtion for basic threshhold for opencv
void hsvConversion(Mat src,int value); //convert each pixel to hsv
double GLCM(Mat &img,Mat &clrImg); //calculate the GLCM
void getGlcm(Mat &img, Mat &src,int value); //creating patches for  GLCM 

void joinmatrix(Mat &src);

int main()
{
	Mat src_gray;
	string imgname;
	double value;


	cout << "Welcome to Corrosion detector by pirashanth!" << endl;                //user display
	cout << "There are image names given below and type one of them!" << endl;                //user display
	cout << "image1.jpg" << endl;
	cout << "image2.jpg" << endl;
	cout << "image3.jpg" << endl;
	cout << "image4.jpg" << endl;
	cout << "image5.jpg" << endl;
	cout << "image6.jpg" << endl;
	cout << "image7.jpg" << endl;
	cout << "image8.jpg" << endl;
	cout << "image9.jpg" << endl;
	cout << "Enter the Image name : (Example :- image1.jpg)" << endl;
	cin >> imgname;													 // user input
	string imageName("../data/" + imgname);
	cout << "Enter the Threshold value : (Example :- 0.05)" << endl;
	cin >> mythreshold;
	src = imread(imageName.c_str(), 1);				//read the image
	if (src.empty())
	{
		cerr << "No image found in the folder ..." << endl;
		return -1;
	}
	/// Applying Median blur
	medianBlur(src, src, 3);
	totrow = src.rows;
	totcol = src.cols;
	totalArea = totrow * totcol;
	cvtColor(src, src_gray, COLOR_BGR2GRAY); //convert input image to Gray scale image for getting GLCM
	getGlcm(src_gray, src,1); 
	//src_cpy = src.clone();
	float corrodedpercentage = corrodedArea * 100 / totalArea;
	cout << "Corrotion Percentage :" << corrodedArea * 100 << "/"<<totalArea << endl;

	/*namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", src);*/
	waitKey(0);
	return 0;

}

void hsvConversion(Mat patchsrc,int value) {    //function for the HSV colour convertion of each pixels of the selected patches from GLCM
	Mat hsv(patchsrc.rows, patchsrc.cols, patchsrc.type());
	if (value == 1) {
		int total_corrosion = 0;
		int corrosionPercentage = 0;
		float r, g, b, h = 0.0f, s, in, v;
		int minv = 50, minsat = 50, maxv = 200;

		int huebins = 180, satbins = 256;
		int sizeOfHist[] = { huebins, satbins };
		// hue varies from 0 to 179, see cvtColor
		float hueranges[] = { 0, 180 };
		// saturation varies from 0 (black-gray-white) to
		// 255 (pure spectrum color)
		float satranges[] = { 0, 256 };
		const float* mainranges[] = { hueranges, satranges };
		MatND histMat;
		// we compute the histogram from the 0-th and 1-st channels
		int histchannels[] = { 0, 1 };
		double value;
		calcHist(&patchsrc, 1, histchannels, Mat(), // do not use mask
			histMat, 2, sizeOfHist, mainranges,
			true, // the histogram is uniform
			false);
		double maxValue = 0;
		minMaxLoc(histMat, 0, &maxValue, 0, 0);



		for (int i = 0; i < patchsrc.rows; i++)
		{
			for (int j = 0; j < patchsrc.cols; j++)
			{
				corrodedArea++;
				b = patchsrc.at<Vec3b>(i, j)[0];
				g = patchsrc.at<Vec3b>(i, j)[1];
				r = patchsrc.at<Vec3b>(i, j)[2];//RGB values of the each pixels

				in = (b + g + r) / 3;			//calculation of I 

				int min_val = 0;
				min_val = std::min(r, std::min(b, g));
				v = min_val;					//calcuation of the value
				s = 1 - 3 * (min_val / (b + g + r));	//saturation value
				//cout << "computed s is :" << s << endl;

				if (s < 0.00001)
				{
					s = 0;
				}
				else if (s > 0.99999) {
					s = 1;
				}
				total_corrosion = s + total_corrosion;
				if (s != 0)
				{
					h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g)*(r - g)) + ((r - b)*(g - b)));
					h = acos(h);		//Hue value

					if (b <= g)
					{
						h = h;
					}
					else {
						h = ((360 * 3.14159265) / 180.0) - h;
					}
				}
				//if pixel black or white those are not corroded pixels
				if (v < minv || v>maxv&&s< minsat) {//avoiding the black and white pixels.
					hsv.at<Vec3b>(i, j)[2] = b;
					hsv.at<Vec3b>(i, j)[1] = g;
					hsv.at<Vec3b>(i, j)[0] = r;
					corrodedArea--;
				}
				else {//rest of the pixels considered as corroioded pixels and sending it for the HS Histogram value		
					float binVal = histMat.at<float>(i, j);
					int intensity = cvRound(binVal * 255 / maxValue);
					
					/*hsv.at<Vec3b>(i, j)[2] = b;
					hsv.at<Vec3b>(i, j)[1] = g;
					hsv.at<Vec3b>(i, j)[0] = r;*/
					if ( binVal>0) {
						hsv.at<Vec3b>(i, j)[2] = (h * 180) / 3.14159265;
						hsv.at<Vec3b>(i, j)[1] = s * 100;
						hsv.at<Vec3b>(i, j)[0] = in;
						//cout << binVal << endl;
						
					}
					else {
						hsv.at<Vec3b>(i, j)[2] = b;
						hsv.at<Vec3b>(i, j)[1] = g;
						hsv.at<Vec3b>(i, j)[0] = r;
						corrodedArea--;
					}
				}
			}
		}
		patchMat = hsv;//replacing HSV patch with the previous patch
	}
	else {
		patchMat = patchsrc; //rest of them became as it was
	}

	waitKey(0);
	joinmatrix(patchMat);  //fuction for joining every patches together
}

void thresh_callback(int, void*)
{
	threshold(src, imBin, thresh, max_thresh, THRESH_BINARY);
	imshow("Result", imBin);
}

double  GLCM(Mat &img,Mat &clrimg)
{
	double energy = 0;
	int row = img.rows, col = img.cols;
	Mat gl = Mat::zeros(256, 256, CV_32F); //creating Mat with 0 value

	//creating glcm matrix with 256 levels,radius=1 and in the horizontal direction 
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col - 1; j++)
			gl.at<float>(img.at<uchar>(i, j), img.at<uchar>(i, j + 1)) = gl.at<float>(img.at<uchar>(i, j), img.at<uchar>(i, j + 1)) + 1;

	// normalizing glcm matrix for parameter determination
	gl = gl + gl.t();
	gl = gl / sum(gl)[0];


	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
		{
			energy = energy + gl.at<float>(i, j)*gl.at<float>(i, j);            //finding energy parameter 
		}
	energy = sqrt(energy);
	if (energy < mythreshold) { //if the energy is lesser than threshhold value  it will forword into HSV coverstion colour
		hsvConversion(clrimg,1);
	}
	else {

		hsvConversion(clrimg, 0);//else it will go in the HSV function and it will never be run as HSV covertion 
	}
	waitKey();
	return energy;
}

void getGlcm(Mat &img,Mat &srcImg,int value) { //it is a function for creating patches for the each loop
	Mat displayImg = img.clone();
	cv::Mat dst;
	if (value == 1) {
		int row = img.rows, col = img.cols;
		Mat dest(row, col, CV_64F);
		int location[1024][2];

		for (int y = 16; y < row - 16; y += 32) { 
			for (int x = 16; x < col - 16; x += 32) {
				Mat subimg = Mat(img, Rect(x - 16, y - 16, 32, 32));  //take a 32x32 subimage
				Mat clrImg = Mat(srcImg, Rect(x - 16, y - 16, 32, 32));
				double feature = GLCM(subimg, clrImg);  //get the energy (or other feature) for this window
				dest.at<double>(y, x) = feature;
				cv::transpose(finalMat, dst);//transposing a  finalMat patch to another Mat
				cv::flip(dst, dst, -1); //flipping that matrix for joining into a image

			}
			s = 0;
			test.push_back(dst); //the patch is pusing inti test matrix
			dst.release();//after it got pushed it will become empty
			finalMat.release();//also final Mat patch will become empty too
		}
		Mat huvMat;
		//cv::transpose(test, huvMat);
		flip(test, huvMat, +1);
		imshow("Rust image", huvMat);

	}
}


void joinmatrix(Mat &src) {
	Mat myflip; // flip the matrix inorder to push the matrix. 
	cv::transpose(src, myflip);
	cv::flip(myflip, myflip, 1);
	if (s < totcol / 32) {
		finalMat.push_back(myflip);//push back function only will allow to push bottom of the matix in order to solve the issue i fliped in and pushing it
		s++;
	}
}
