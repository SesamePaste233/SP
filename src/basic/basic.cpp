#include "basic_header.h"
using namespace cv;
using namespace std;

Mat basic::RAW2MAT1(string src,int R,int C,int b,string type,int band){
	FILE* fp=fopen(src.c_str(),"rb");
	if(fp==NULL){
		error;
	}
	uchar* Rawdata=new uchar[R*C*b*sizeof(uchar)];
	if(Rawdata==NULL){
		error;
	}
	if(band<=0||band>b){
		error;
	}
	fread(Rawdata,sizeof(uchar),R*C*b,fp);
	Mat img(R,C,CV_8UC1,Scalar(0,0,0));
	uchar* ptr=img.data;
	int nChannels=img.channels();
	int nRows=img.rows;
	int nCols=img.cols;
	if(type=="BSQ"){
		for(int i=0;i<nChannels;i++){
			for(int r=0;r<nRows;r++){
				for(int c=0;c<nCols;c++){
					ptr[(r*nCols+c)*nChannels+i]=Rawdata[(r+nRows*(band-1))*nCols+c];
				}
			}
		}
	}
	else if(type=="BIL"){
		for(int r=0;r<nRows;r++){
			for(int i=0;i<nChannels;i++){
				for(int c=0;c<nCols;c++){
					ptr[(r*nCols+c)*nChannels+i]=Rawdata[(r*b+(band-1))*nCols+c];
				}
			}
		}
	}
	else if(type=="BIP"){
		for(int r=0;r<nRows;r++){
			for(int c=0;c<nCols;c++){
				for(int i=0;i<nChannels;i++){
					ptr[(r*nCols+c)*nChannels+i]=Rawdata[(r*nCols+c)*b+band-1];
				}
			}
		}
	}
	else error;
	delete[] Rawdata;
	fclose(fp);
	return img;
}

Mat basic::PixelProcess(Mat img,function<uchar (uchar)>* func){
	Mat oimg=img.clone();
	uchar* ptr=oimg.data;
	int nChannels=oimg.channels();
	int nRows=oimg.rows;
	int nCols=oimg.cols;

	for(int i=0;i<nChannels;i++){
		for(int r=0;r<nRows;r++){
			for(int c=0;c<nCols;c++){
				ptr[(r*nCols+c)*nChannels+i]=func[i](ptr[(r*nCols+c)*nChannels+i]);
			}
		}
	}
	return oimg;
}

Mat basic::KernelProcess(Mat simg,convKernel3x3 kernel,bool DEBUG){
	Mat iimg=simg.clone();
	int rows=iimg.rows;
	int cols=iimg.cols;
	int n=iimg.channels();
	int type;
	if(!DEBUG){
		if(n==1){
			type=CV_8UC1;
		}else if(n==3){
			type=CV_8UC3;
		}
	}else{
		if(n==1){
			type=CV_32S;
		}
	}
	Mat img(rows+2,cols+2,type,Scalar(0,0,0));
	uchar* ptr=img.data;
	int nChannels=img.channels();
	int nRows=img.rows;
	int nCols=img.cols;

	for(int i=0;i<nChannels;i++){
		for(int r=0;r<nRows;r++){
			for(int c=0;c<nCols;c++){
				if(DEBUG){
					if(r==0||r==nRows-1||c==0||c==nCols-1){
						img.at<int>(r,c)=0;
					}else{
						img.at<int>(r,c)=iimg.data[((r-1)*cols+c-1)*nChannels+i];
					}
					continue;
				}
				if(r==0||r==nRows-1||c==0||c==nCols-1){
					ptr[(r*nCols+c)*nChannels+i]=0;
				}else{
					ptr[(r*nCols+c)*nChannels+i]=iimg.data[((r-1)*cols+c-1)*nChannels+i];
				}
			}
		}
	}

	Mat Img=img.clone();
	for(int i=0;i<nChannels;i++){
		for(int r=1;r<nRows-1;r++){
			for(int c=1;c<nCols-1;c++){
				int sum=0;
				for(int ii=1-kernel.anchor.y;ii<=3-kernel.anchor.y;ii++)
					for(int jj=1-kernel.anchor.x;jj<=3-kernel.anchor.x;jj++){
						if(DEBUG){
							sum+=img.at<int>(r+ii,c+jj)*kernel.k(ii,jj);
							continue;
						}
						sum+=img.data[((r+ii)*nCols+c+jj)*nChannels+i]*kernel.k(ii,jj);
					}
				if(DEBUG)
					Img.at<int>(r,c)=kernel.processor(kernel.multiplier*sum);
				else
					Img.data[(r*nCols+c)*nChannels+i]=kernel.processor(kernel.multiplier*sum);
			}
		}
	}

	Mat oimg(Img,Range(1,rows+1),Range(1,cols+1));
	return oimg;
}

Mat basic::ImgTransform(Mat simg,transformOptions opt){
	Mat img=simg.clone();
	int type;
	int nChannels=img.channels();
	int nRows=img.rows;
	int nCols=img.cols;
	if(nChannels==1){
		type=CV_8UC1;
	}else if(nChannels==3){
		type=CV_8UC3;
	}
	uchar* ptr=img.data;
	Mat iimg(nRows,nCols,type,Scalar(0,0,0));
	//rotate&translate
	if(!(opt.x_translation==0&&opt.y_translation==0&&opt.rot_angle==0))
	for(int i=0;i<nChannels;i++){
		for(int r=0;r<nRows;r++){
			for(int c=0;c<nCols;c++){
				int R=(r-nRows/2)*cos(CV_PI*opt.rot_angle/180.0)-(c-nCols/2)*sin(CV_PI*opt.rot_angle/180.0)-opt.y_translation+nRows/2;
				int C=(r-nRows/2)*sin(CV_PI*opt.rot_angle/180.0)+(c-nCols/2)*cos(CV_PI*opt.rot_angle/180.0)-opt.x_translation+nCols/2;
				if(R>=nRows||R<0||C>=nCols||C<0)
					iimg.data[(r*nCols+c)*nChannels+i]=0;
				else
					iimg.data[(r*nCols+c)*nChannels+i]=img.data[(R*nCols+C)*nChannels+i];
			}
		}
	}
	else iimg=img;
	//resize
	int rows=abs(opt.r_size*nRows);
	int cols=abs(opt.c_size*nCols);
	if(rows==0||cols==0)error;
	if(rows==nRows&&cols==nCols)return iimg;
	Mat oimg(rows,cols,type,Scalar(0,0,0));
	for(int i=0;i<nChannels;i++){
		for(int r=0;r<rows;r++){
			for(int c=0;c<cols;c++){
				int R=r*nRows/(double)rows;
				int C=c*nCols/(double)cols;
				oimg.data[(r*cols+c)*nChannels+i]=iimg.data[(R*nCols+C)*nChannels+i];
			}
		}
	}
	return oimg;
}

Mat basic::ImgStack(Mat R,Mat G,Mat B){
	Mat img(R.rows,R.cols,CV_8UC3,Scalar(0,0,0));
	uchar* ptr=img.data;
	int nChannels=img.channels();
	int nRows=img.rows;
	int nCols=img.cols;
	for(int r=0;r<nRows;r++){
		for(int c=0;c<nCols;c++){
			ptr[(r*nCols+c)*3+0]=R.data[r*nCols+c];
		}
	}
	for(int r=0;r<nRows;r++){
		for(int c=0;c<nCols;c++){
			ptr[(r*nCols+c)*3+1]=G.data[r*nCols+c];
		}
	}
	for(int r=0;r<nRows;r++){
		for(int c=0;c<nCols;c++){
			ptr[(r*nCols+c)*3+2]=B.data[r*nCols+c];
		}
	}
	return img;
}

vector<int> basic::extractEOHFeatures(cv::Mat iimg){
	Mat dx=basic::KernelProcess(iimg,basic_Kernels::Sobel("horizontal"),true);
	Mat dy=basic::KernelProcess(iimg,basic_Kernels::Sobel("vertical"),true);
	int hist[37]={0};
	Mat theta(dx.rows,dx.cols,CV_32S);
	for(int r=0;r<theta.rows;r++){
		for(int c=0;c<theta.cols;c++){
			int dX=dx.at<int>(r,c);
			int dY=dy.at<int>(r,c);
			int& data=theta.at<int>(r,c);
			if(dX==0){
				if(dY>0)
					data=90;
				else if(dY<0)
					data=-90;
				else
					data=95;
			}
			else
				data=180/CV_PI*atan(dY/(double)dX);
			hist[(unsigned int)(((data+90.0)-0.5)/5.0)]++;
		}
	}
	vector<int> Hist;
	for(int i=0;i<36;++i){Hist.push_back(hist[i]);};
	return Hist;
}

void basic::printHist(string name,vector<int> hist,int data_lenght,int size){
	int Size=0;
	if(data_lenght==0)
		Size=hist.size();
	else
		Size=data_lenght;
	Mat histImg(size,Size,CV_8UC1,Scalar(0));
	int hpt=(int)(0.9*size);
	int max=0;
	for(int i=0;i<Size;i++){
		if(hist[i]>max)max=hist[i];
	}
	for(int i=0;i<Size;i++)
	{
		float binVal=hist[i]/(float)max;
		if(binVal>0)
		{
			int intensity=(int)(binVal*hpt);
			line(histImg,cv::Point(i,size),
				Point(i,size-intensity),
				Scalar(255),1);
		}
	}
	Mat oimg=basic::ImgTransform(histImg,transformOptions(1,size/Size,0,0,0));
	imshow(name,oimg);
}

Mat basic::rgb2gray(Mat rgbimg){
	int nChannels=rgbimg.channels();
	int rows=rgbimg.rows;
	int cols=rgbimg.cols;
	Mat oimg(rows,cols,CV_8UC1,Scalar(0,0,0));
	for(int r=0;r<rows;r++){
		for(int c=0;c<cols;c++){
			int sum=0;
			for(int i=0;i<nChannels;i++){
				sum+=rgbimg.data[(r*cols+c)*nChannels+i];
			}
			oimg.data[r*cols+c]=sum/3.0;
		}
	}
	return oimg;
}

vector<Point3i> basic::searchCircles(Mat aimg,Range threshold,int accu,int cent,int r_max,bool DEBUG){
	if(r_max==0){
		r_max=(aimg.cols>aimg.rows?aimg.rows:aimg.cols)/2;
	}

	/*预处理*/

	//图像二值化
	Mat kimg;
	function<uchar (uchar)> Binarize=[threshold](uchar a)->uchar{
		if(threshold.start<=a&&a<=threshold.end){
			return 255;
		}else return 0;
	};
	if(aimg.type()==CV_8UC3){
		kimg=basic::rgb2gray(aimg);
		kimg=basic::PixelProcess(kimg,&Binarize);
	}
	else if(aimg.type()==CV_8UC1){
		kimg=basic::PixelProcess(aimg,&Binarize);
	}
	//边缘点提取 Sobel算子
	Mat iimg=basic::KernelProcess(kimg,basic_Kernels::Gaussian());
	Mat dx=basic::KernelProcess(iimg,basic_Kernels::Sobel("horizontal"),true);
	Mat dy=basic::KernelProcess(iimg,basic_Kernels::Sobel("vertical"),true);
	int rows=dx.rows;
	int cols=dx.cols;

	/*寻找圆心*/

	//生成权值图 counter
	Mat counter(rows,cols,CV_32S,Scalar(0));
	for(int r=0;r<rows;r++){
		for(int c=0;c<cols;c++){
			int dX=dx.at<int>(r,c);
			int dY=dy.at<int>(r,c);
			if(dX!=0||dY!=0){
				if(dX==0||abs(dY/dX)>1){
					for(int i=0;;i++){
						if(i*i*(dX*dX+dY*dY)>r_max*r_max*dY*dY||r+i>=rows||c+i*dX/dY>=cols||r+i<0||c+i*dX/dY<0)break;
						counter.at<int>(r+i,c+i*dX/dY)+=1;
					}
					for(int i=1;;i++){
						if(i*i*(dX*dX+dY*dY)>r_max*r_max*dY*dY||r-i>=rows||c-i*dX/dY>=cols||r-i<0||c-i*dX/dY<0)break;
						counter.at<int>(r-i,c-i*dX/dY)+=1;
					}
				}
				else if(abs(dY/dX)<=1){
					for(int i=0;;i++){
						if(i*i*(dX*dX+dY*dY)>r_max*r_max*dX*dX||r+i*dY/dX>=rows||c+i>=cols||r+i*dY/dX<0||c+i<0)break;
						counter.at<int>(r+i*dY/dX,c+i)+=1;
					}
					for(int i=1;;i++){
						if(i*i*(dX*dX+dY*dY)>r_max*r_max*dX*dX||r-i*dY/dX>=rows||c-i>=cols||r-i*dY/dX<0||c-i<0)break;
						counter.at<int>(r-i*dY/dX,c-i)+=1;
					}
				}
			}
		}
	}
	//显示过程中的图像
	if(DEBUG){
		Mat dimg=basic::convert2Img(counter);
		imshow("img",dimg);//权值图
		imwrite("Debug\\weight-img.jpg",dimg);
		imshow("dX",basic::convert2Img(dx));//X轴梯度
		imshow("dY",basic::convert2Img(dy));//Y轴梯度
	}
	//根据权值图寻找圆心
	stack<Point3i> centers;
	centers.push(Point3i(0,0,1));
	for(int r=2;r<rows-2;r++){
		for(int c=2;c<cols-2;c++){
			int count=0;
			for(int ii=-2;ii<=2;ii++){
				for(int jj=-2;jj<=2;jj++){
					count+=counter.at<int>(r+ii,c+jj);
				}
			}
			if(cent?(count>cent):(count>centers.top().z))centers.push(Point3i(c,r,count));
		}
	}

	/*计算半径*/

	vector<Point3i> circles;//储存有圆的圆心坐标，和半径
	for(int ii=0;ii<centers.size();ii++){
		if(centers.empty())break;
		Point3i center=centers.top();
		centers.pop();
		//去除过近的圆心
		for(;!centers.empty()&&sqrt((float)((centers.top().y-center.y)*(centers.top().y-center.y)+(centers.top().x-center.x)*(centers.top().x-center.x)))<accu;){
			centers.pop();
		}
		//生成半径直方图
		int r_count[2048]={0};
		for(int r=0;r<rows;r++){
			for(int c=0;c<cols;c++){
				int dX=dx.at<int>(r,c);
				int dY=dy.at<int>(r,c);
				if(dX!=0||dY!=0){
					int rr=sqrt((float)((r-center.y)*(r-center.y)+(c-center.x)*(c-center.x)))/accu;
					r_count[rr]++;
				}
			}
		}
		//寻找边缘点聚集的圆环内径 R
		float m_ratio=0.5;
		int R=0;
		for(int i=1;i<2048;i++){
			float ratio=r_count[i]/((2*i*accu+accu*accu)*CV_PI);
			if(ratio>m_ratio)m_ratio=ratio,R=i;
		}
		//得到半径
		center.z=R*accu;
		if(center.z!=0)circles.push_back(center);
	}
	return circles;
}

Mat basic::convert2Img(Mat mat){
	int rows=mat.rows,cols=mat.cols;
	Mat img(mat.rows,mat.cols,CV_8UC1,Scalar(0));

	function<uchar (int)>f=[](int a)->uchar{
		if(a<0)return 0;
		else if (a>255)return 255;
		else return a;
	};

	for(int r=0;r<rows;r++){
		for(int c=0;c<cols;c++){
			img.data[r*cols+c]=f(mat.at<int>(r,c));
		}
	}
	return img;
}

cv::Mat basic::edges(cv::Mat shapes,Scalar bgr,Range threshold){
	Mat kimg;
	function<uchar (uchar)> Binarize=[threshold](uchar a)->uchar{
		if(threshold.start<=a&&a<=threshold.end){
			return 255;
		}else return 0;
	};
	if(shapes.type()==CV_8UC3){
		kimg=basic::rgb2gray(shapes);
		kimg=basic::PixelProcess(kimg,&Binarize);
	}
	else if(shapes.type()==CV_8UC1){
		kimg=basic::PixelProcess(shapes,&Binarize);
	}
	Mat iimg=basic::KernelProcess(kimg,basic_Kernels::Laplacian1());
	Mat oimg=shapes.clone();
	for(int r=0;r<iimg.rows;r++){
		for(int c=0;c<iimg.cols;c++){
			if(iimg.at<uchar>(r,c)==255){
				for(int i=0;i<3;i++){
					oimg.data[(r*oimg.cols+c)*3+i]=bgr[i];
				}
			}
		}
	}
	return oimg;
}