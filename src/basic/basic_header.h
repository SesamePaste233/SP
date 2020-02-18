#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <stdarg.h>
#include <stack>
class convKernel3x3{//convolution kernel 3x3
public:
	cv::Mat_<double> kernel;
	cv::Point anchor;
	double multiplier;
	std::function<double (double)> processor;
	convKernel3x3():anchor(2,2),processor(Default),multiplier(1){};
	double k(int i,int j){
		int r=anchor.y-1+i;
		int c=anchor.x-1+j;
		return kernel.at<double>(r,c);
	}
private:
	static double Default(double a){return a;};
	
};


class transformOptions{
public:
	double r_size;
	double c_size;
	double rot_angle;
	int x_translation;
	int y_translation;
	transformOptions(double r=1,double c=1,int dx=0,int dy=0,double a=0){
		r_size=r,c_size=c,x_translation=dx,y_translation=dy,rot_angle=a;
	};
};

namespace basic{
//Read images from RAW files.
cv::Mat RAW2MAT1(std::string ifs,int nRows,int nCols,int nbands,std::string type="BSQ",int band=1);

//Process image based on pixels.
cv::Mat PixelProcess(cv::Mat img,std::function<uchar (uchar)>* func);

//Process image based on windows.
cv::Mat KernelProcess(cv::Mat img,convKernel3x3 kernel,bool DEBUG=false);

//Rotate, resize, translate images.
cv::Mat ImgTransform(cv::Mat img,transformOptions opt);

//Stack gray-images into rgb-image.
cv::Mat ImgStack(cv::Mat B,cv::Mat G,cv::Mat R);

//Extract EOH features vector.
std::vector<int> extractEOHFeatures(cv::Mat img);

//Draw features vector.
void printHist(std::string name,std::vector<int> hist,int data_lenght=0,int size=240);

//Turn rgb-image into gray-image.
cv::Mat rgb2gray(cv::Mat rgbimg);

//Search for circles in image.
std::vector<cv::Point3i> searchCircles(cv::Mat aimg,cv::Range threshold,int accu,int cent=0,int r_max=0,bool DEBUG=false);

//Convert Mat_<int> into Mat_<uchar>.
cv::Mat convert2Img(cv::Mat mat);

//show edges of a image.
cv::Mat edges(cv::Mat shapes,cv::Scalar bgr,cv::Range threshold=cv::Range(128,255));
}

namespace basic_Kernels{//frequently used convolution kernels.
	static convKernel3x3 Laplacian1(uchar T=1){
		convKernel3x3 K;
		K.kernel=(cv::Mat_<double>(3,3)<<-1,-1,-1,-1,8,-1,-1,-1,-1);
		K.multiplier=1/8.0;
		K.processor=[T](double a)->double{
			if(abs(a)<T)
				return 0;
			else return 255;
		};
		return K;
	}

	static convKernel3x3 Laplacian2(){
		convKernel3x3 K;
		K.kernel=(cv::Mat_<double>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
		K.multiplier=1/5.0;
		return K;
	}

	static convKernel3x3 MeanFilter(bool center=0){
		convKernel3x3 K;
		K.kernel=(cv::Mat_<double>(3,3)<<1,1,1,1,(int)center,1,1,1,1);
		K.multiplier=1/(double)(8+(int)center);
		return K;
	}

	static convKernel3x3 Gaussian(){
		convKernel3x3 K;
		K.kernel=(cv::Mat_<double>(3,3)<<0.0947416,0.118318,0.0947416,0.118318,0.147761,0.118318,0.0947416,0.118318,0.0947416);
		return K;
	}

	static convKernel3x3 Sobel(std::string type,bool direction=true){
		convKernel3x3 K;
		if(type=="vertical")
			K.kernel=(cv::Mat_<double>(3,3)<<0,-1,0,0,0,0,0,1,0);
		else if(type=="horizontal")
			K.kernel=(cv::Mat_<double>(3,3)<<0,0,0,-1,0,1,0,0,0);
		K.multiplier=(direction?1:-1);
		return K;
	}

}