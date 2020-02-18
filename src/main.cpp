#include <basic_header.h>
using namespace std;
using namespace cv;

void main(){
	Mat img=imread("ImgInput\\src.jpg",cv::IMREAD_COLOR);
	ifstream opts("Options.txt");
	string s;
	bool Debug=false;
	while(getline(opts,s)){
		if(s.compare("Debug=1")==0||s.compare("Debug=true"))
			Debug=true;
	}
	vector<Point3i> circles=basic::searchCircles(img,Range(128,255),10,0,0,Debug);//寻找圆形
	Mat oimg=img.clone();
	if(img.type()==CV_8UC1){
		Mat iimg=basic::ImgStack(img,img,img);
		oimg=iimg.clone();
	}
	for(int i=0;i<1/*circles.size()*/;i++){//画圆形
		circle(oimg,Point(circles[i].x,circles[i].y),circles[i].z,Scalar(0,0,255),5);
		circle(oimg,Point(circles[i].x,circles[i].y),2,Scalar(0,0,255),2);
	}
	imshow("image",oimg);
	cout<<"Showing images..."<<endl<<"Close image windows to continue.";
	imwrite("ImgOutput\\result.jpg",oimg);//保存结果
}