#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>
#include <string>
#include <time.h>
#include <io.h>
#include <iomanip>
#include <mmintrin.h>
#include <sstream>
using namespace cv;
using namespace std;
#define PI 3.141592653f

void myGaborKernel(float theta, Mat &oddKernel, Mat &evenKernel)
{
	float lamda = 5.0f;
	int size_kernel = 17;
	float sigma = (float)size_kernel / 9.0f;
	int k = (size_kernel - 1) / 2;
	for (int x = -1 * k; x <= k; x++)
		for (int y = -1 * k; y <= k; y++)
		{
			float a = x*cos(theta) + y*sin(theta);
			float b = y*cos(theta) - x*sin(theta);
			float oddResp = exp(-1.0f / 8.0f / sigma / sigma*(4.0f*a*a + b*b)) * sin(2.0f*PI*a / lamda);
			float evenResp = exp(-1.0f / 8.0f / sigma / sigma*(4.0f*a*a + b*b)) * cos(2.0f*PI*a / lamda);
			oddKernel.at<float>(x + k , y + k) = oddResp;
			evenKernel.at<float>(x + k , y + k) = evenResp;
		}

	float u1 = mean(oddKernel)[0];
	float u2 = mean(evenKernel)[0];
	float *f1 = (float*)oddKernel.datastart;
	float *f2 = (float*)oddKernel.dataend;
	Mat_<float>::iterator oddit1 = oddKernel.begin<float>(), oddit2 = oddKernel.end<float>();
	Mat_<float>::iterator evenit1 = evenKernel.begin<float>(), evenit2 = evenKernel.end<float>();

	for (int i = 0; i < size_kernel; i++)
	{
		for (int j = 0; j < size_kernel; j++)
		{
			oddKernel.at<float>(i, j) = oddKernel.at<float>(i, j) - u1;
			evenKernel.at<float>(i, j) = evenKernel.at<float>(i, j) - u2;
		}
	}
	
	//get the L2Norm
	float l2sum1 = 0, l2sum2 = 0;
	for_each(oddit1, oddit2, [&l2sum1](float x){ l2sum1 += x*x; });
	for_each(evenit1, evenit2, [&l2sum2](float x){ l2sum2 += x*x; });
	l2sum1 /= (17.0f*17.0f);
	l2sum2 /= (17.0f*17.0f);
	//divide the L2Norm
	for (int i = 0; i < size_kernel; i++)
		for (int j = 0; j < size_kernel; j++)
		{
			oddKernel.at<float>(i, j) = oddKernel.at<float>(i, j) / l2sum1;
			evenKernel.at<float>(i, j) = evenKernel.at<float>(i, j) / l2sum2;
		}
	//for (int i = 0; i < size_kernel; i++)
	//{
	//	for (int j = 0; j < size_kernel; j++)
	//		cout << setw(8) << setiosflags(ios::fixed) << setprecision(2) << oddKernel.at<float>(i, j) << ' ';
	//	cout << endl;
	//}
	//cout << endl;
	//for (int i = 0; i < size_kernel; i++)
	//{
	//	for (int j = 0; j < size_kernel; j++)
	//		cout << setw(8) << setiosflags(ios::fixed) << setprecision(2) << evenKernel.at<float>(i, j) << ' ';
	//	cout << endl;
	//}

	//for_each(oddit1, oddit2, [&l2sum1](float x){ x /= l2sum1; });
	//for_each(evenit1, evenit2, [&l2sum2](float x){ x /= l2sum2; });
}

void visit(string path, Mat (*func)(string))
{
	struct _finddata_t   filefind;
	string  curr = path + "\\*.*";
	int   done = 0, handle;
	if ((handle = _findfirst(curr.c_str(), &filefind)) == -1)return;
	while (!(done = _findnext(handle, &filefind)))
	{
		printf("%s\n", filefind.name);
		if (!strcmp(filefind.name, "..")){
			continue;
		}
		
		if ((_A_SUBDIR == filefind.attrib)) //是目录  
		{
			printf("----------%s\n", filefind.name);
			cout << filefind.name << "(dir)" << endl;
			curr = path + "\\" + filefind.name;
		}
		else//是文件       
		{
			time_t start, stop;
			string filePath = path + "\\" + filefind.name;
			cout << "processing...." << filePath << '.';

			start = clock();
			Mat img = func(filePath);
			stop = clock();
			printf("Use Time:%ld\n", (stop - start) * 1000 / CLOCKS_PER_SEC);
			cout << path + "\\results\\" + filefind.name << endl;
			imwrite(path + "\\results\\" + filefind.name, img);
		}
		
	}
	_findclose(handle);
}

Mat computeVpScore(string filePath)
{
	Mat image_origin = imread(filePath);
	Mat img_gray, img_float;
	cvtColor(image_origin, img_gray, CV_RGB2GRAY);
	img_gray.convertTo(img_float, CV_32F);

	int n_theta = 36;
	int width = 128;
	float scale_factor = (float)width / (float)image_origin.cols;
	Mat image(img_gray.rows * scale_factor, width, CV_32F);
	resize(img_float, image, image.size());
	int m = image.rows, n = image.cols;
	cout << m << ' ' << n << endl;
	Mat scores(m, n, CV_32F);
	float ***gabors = new float**[m];
	for_each(gabors, gabors+m, [n](float** &x){x = new float*[n]; });
	for_each(gabors, gabors+m, [n, n_theta](float** x){for_each(x, x+n, [&, n_theta](float* &y){ y = new float[n_theta]; }); });
	//cout << image;
	waitKey(100000);
	namedWindow("g");
	namedWindow("o");
	namedWindow("e");
	for (int t = 0; t < n_theta; t++)
	{	
		float theta = PI*(float)t / (float)n_theta;
		Mat oddKernel(17, 17, CV_32F), evenKernel(17, 17, CV_32F);
		myGaborKernel(theta, oddKernel, evenKernel);
		Mat filtered(image.rows, image.cols, CV_32F), oddfiltered(image.rows, image.cols, CV_32F), evenfiltered(image.rows, image.cols, CV_32F);
		filter2D(image, oddfiltered, -1, oddKernel);
		filter2D(image, evenfiltered, -1, evenKernel);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				filtered.at<float>(i, j) = pow(oddfiltered.at<float>(i, j), 2.0f) + pow(evenfiltered.at<float>(i, j), 2.0f);
				//filtered.at<float>(i, j) = oddfiltered.at<float>(i, j) + evenfiltered.at<float>(i, j);
		
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				gabors[i][j][t] = filtered.at<float>(i, j);
	}
	destroyAllWindows();
	
	Mat directions(image.rows, image.cols, CV_8U);
	Mat confidences(image.rows, image.cols, CV_32F);
	
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		{

			int idx = (float)(max_element(gabors[i][j], gabors[i][j] + n_theta) - gabors[i][j]);
			directions.at<uchar>(i, j) = (uchar)idx;
			float max_resp = gabors[i][j][idx];
			sort(gabors[i][j], gabors[i][j] + n_theta, greater<float>());
			if (max_resp > 0.5f)
				confidences.at<float>(i, j) = (1 - accumulate(gabors[i][j] + 4, gabors[i][j] + 15, 0.0f) / 11.0f / max_resp);
			else
				confidences.at<float>(i, j) = 0;
		}
	
	float thresh = 2.0f * 180.0f / (float)n_theta;
	std::cout << thresh << endl;
	int r = (m + n) / 7;
	float r_dia = sqrtf(m*m + n*n);
	for (int i = 0; i < m; i++)
	{	
		for (int j = 0; j < n; j++)
		{
			scores.at<float>(i, j) = 0;
			for (int i1=i+1; i1 < m && i1<i+40; i1++)
			{
				for (int j1=0; j1 < n; j1++)
				{
					float c = (float)directions.at<uchar>(i1, j1) / (float)n_theta * 180;
					if (c < 5.0f || (85.0f < c && c < 95.0f) || c>175.0f)
						continue;
					float d = sqrtf(pow(i - i1, 2.0) + pow(j - j1, 2.0));
					float gamma = acosf(((float)j - (float)j1) / d)/ PI * 180.0f;					
					if (abs(c - gamma) < thresh && confidences.at<float>(i1, j1) > 0.35)
					{
						scores.at<float>(i, j) = scores.at<float>(i, j) + 1 / (1 + pow(c - gamma, 2.0f)*pow(d / r_dia, 2.0));
					}	
				}
			}
		}
	}

	Point p_max, p_min;	
	double score_max, score_min;
	cv::minMaxLoc(scores, &score_min, &score_max, &p_min, &p_max);
	float scale = score_max / 255.0f;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			scores.at<float>(i, j) = round(scores.at<float>(i, j) / scale);

	cv::circle(image_origin, cvPoint(p_max.x / scale_factor, p_max.y / scale_factor), 10, Scalar(0), 5, 8, 0);

	//Release memory
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			delete[] gabors[i][j];
	for (int i = 0; i < m; i++)
		delete[] gabors[i];
	delete[] gabors;
	//cout << scores;
	return image_origin;

}

int main()
{
	visit("C:\\images", computeVpScore);

	/*
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	bool stop = false;
	while (!stop)
	{
		cap >> frame;
		imshow("当前视频", frame);
		if (waitKey(30) >= 0)
			stop = true;
	}
	*/
	system("pause");
	return 0;
}