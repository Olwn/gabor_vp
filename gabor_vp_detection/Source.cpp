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

struct vals{
	float arc;
	float d;
};

int ***gabors;
vals arcMatrix[200][400];
Mat oddKernels[36], evenKernels[36];
int directions[300][300];
int confidences[300][300];

void myGaborKernel(float theta, Mat &oddKernel, Mat &evenKernel)
{
	float lamda = 5.0f;
	int size_kernel = 17;
	float sigma = (float)size_kernel / 9.0f;
	int k = (size_kernel - 1) / 2;
	oddKernel.create(size_kernel, size_kernel, CV_32F);
	evenKernel.create(size_kernel, size_kernel, CV_32F);
	for (int x = -1 * k; x <= k; x++)
		for (int y = -1 * k; y <= k; y++)
		{
			float a = x*cos(theta) + y*sin(theta);
			float b = y*cos(theta) - x*sin(theta);
			float oddResp = exp(-1.0f / 8.0f / sigma / sigma*(4.0f*a*a + b*b)) * sin(2.0f*PI*a / lamda);
			float evenResp = exp(-1.0f / 8.0f / sigma / sigma*(4.0f*a*a + b*b)) * cos(2.0f*PI*a / lamda);
			oddKernel.at<float>(x + k, y + k) = oddResp;
			evenKernel.at<float>(x + k, y + k) = evenResp;
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
	//for_each(oddit1, oddit2, [&l2sum1](float x){ x /= l2sum1; });
	//for_each(evenit1, evenit2, [&l2sum2](float x){ x /= l2sum2; });
}

void visit(string path, Mat(*func)(string))
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

		if ((_A_SUBDIR == filefind.attrib)) //ÊÇÄ¿Â¼  
		{
			printf("----------%s\n", filefind.name);
			cout << filefind.name << "(dir)" << endl;
			curr = path + "\\" + filefind.name;
		}
		else//ÊÇÎÄ¼þ       
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

class Parallel_process : public cv::ParallelLoopBody
{

private:
	cv::Mat& scores;

public:
	Parallel_process(cv::Mat& outImage)
		:scores(outImage){}

	virtual void operator()(const cv::Range& range) const
	{
		int m = scores.rows, n = scores.cols;
		//cout << m << ' ' << n << ' ' << endl;
		int num_threads = (range.end - range.start);
		for (int r = 0; r < num_threads; r++)
		{
			for (int i = r*m / num_threads; i < (r + 1)*m / num_threads; i++)
			{
				for (int j = 0; j < n; j++)
				{
					float tmepScore = 0;

					for (int i1 = i + 1; i1 < m && i1<i + 40; i1++)
					{
						for (int j1 = 0; j1 < n; j1++)
						{
							if (i1 % 2 == 0 || j1 % 2 == 0)
								continue;
							int gamma = arcMatrix[i1 - i][j1 - j + 200].arc;
							if (abs(directions[i1][j1] - gamma) < 10 && confidences[i1][j1] > 35)
							{
								//tmepScore += 1 / (1 + pow(c - gamma, 2.0f)*pow(d / r_dia, 2.0));
								tmepScore += 1 / (1 + abs(directions[i1][j1] - gamma) / 2);
								//tmepScore += 1;
							}
						}
					}
					scores.at<float>(i, j) = tmepScore;
				}
			}
		}
	}
};

inline Mat computeVpScore(const Mat &image_origin)
{
	time_t t1, t2;
	t1 = clock();
	Mat img_gray, img_float;
	cvtColor(image_origin, img_gray, CV_RGB2GRAY);
	img_gray.convertTo(img_float, CV_32F);

	int n_theta = 36;
	int width = 128;
	float scale_factor = (float)width / (float)image_origin.cols;
	Mat image = img_float;
	/*Mat image(img_gray.rows * scale_factor, width, CV_32F);
	resize(img_float, image, image.size());*/
	int m = image.rows, n = image.cols;
	
	Mat filtered(image.rows, image.cols, CV_32F), oddfiltered(image.rows, image.cols, CV_32F), evenfiltered(image.rows, image.cols, CV_32F);
	
	for (int t = 0; t < n_theta; t++)
	{
		filter2D(image, oddfiltered, -1, oddKernels[t]);
		filter2D(image, evenfiltered, -1, evenKernels[t]);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				filtered.at<float>(i, j) = abs(oddfiltered.at<float>(i, j)) + abs(evenfiltered.at<float>(i, j));

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				gabors[i][j][t] = filtered.at<float>(i, j);
	}
	
	//Mat directions(image.rows, image.cols, CV_8U);
	//Mat confidences(image.rows, image.cols, CV_32F);

	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		{

			int idx = (float)(max_element(gabors[i][j], gabors[i][j] + n_theta) - gabors[i][j]);
			//directions.at<uchar>(i, j) = (uchar)idx;
			directions[i][j] = (float)idx / (float)n_theta * 180.0f;
			
			float max_resp = gabors[i][j][idx];
			sort(gabors[i][j], gabors[i][j] + n_theta, greater<int>());
			if (max_resp > 0.5f)
				confidences[i][j] = 100*(1 - accumulate(gabors[i][j] + 4, gabors[i][j] + 15, 0.0f) / 11.0f / max_resp);
			else
				confidences[i][j] = 0;
		}
	int thresh = 2.0f * 180.0f / (float)n_theta;
	int r = (m + n) / 7;
	float r_dia = sqrtf(m*m + n*n);
	time_t start, end;
	start = clock();
	Mat scores(m, n, CV_32F);
	parallel_for_(Range(0, 2), Parallel_process(scores));
	//for (int i = 0; i < m; i++)
	//{
	//	for (int j = 0; j < n; j++)
	//	{
	//		scores.at<float>(i, j) = 0;
	//		float tmepScore = 0;
	//		for (int i1 = i + 1; i1 < m && i1<i + 30; i1++)
	//		{
	//			for (int j1 = 0; j1 < n; j1++)
	//			{
	//				if (j1 % 2 == 0 || i1 % 2 == 0)
	//					continue;
	//				//int c = (float)directions.at<uchar>(i1, j1) / (float)n_theta * 180.0f;
	//				/*if (c < 5.0f || (85.0f < c && c < 95.0f) || c>175.0f)
	//				continue;*/
	//				//float d = sqrtf(pow(i - i1, 2.0) + pow(j - j1, 2.0));
	//				//float d = arcMatrix[i1-i][j1-j+200].d;
	//				//float gamma = acosf(((float)j - (float)j1) / d)/ PI * 180.0f;					
	//				int gamma = arcMatrix[i1 - i][j1 - j + 200].arc;

	//				if (abs(directions[i1][j1] - gamma) < thresh && confidences[i1][j1]> 35)
	//				{
	//					//tmepScore += 1 / (1 + pow(c - gamma, 2.0f)*pow(d / r_dia, 2.0));
	//					tmepScore += 1 / (1 + abs(directions[i1][j1] - gamma)/2);
	//					//tmepScore += 1;
	//				}
	//			}
	//		}
	//		scores.at<float>(i, j) = tmepScore;
	//	}
	//}
	end = clock();
	cout << "voting use time of " << (end - start) * 1000 / CLOCKS_PER_SEC<<endl;
	Point p_max, p_min;
	double score_max, score_min;
	cv::minMaxLoc(scores, &score_min, &score_max, &p_min, &p_max);
	//float scale = score_max / 255.0f;
	//for (int i = 0; i < m; i++)
	//	for (int j = 0; j < n; j++)
	//		scores.at<float>(i, j) = round(scores.at<float>(i, j) / scale);

	cv::circle(image_origin, cvPoint(p_max.x / scale_factor, p_max.y / scale_factor), 5, Scalar(0,255,255), 5, 8, 0);
	t2 = clock();
	cout << "sorting use time of " << (t2 - t1) * 1000 / CLOCKS_PER_SEC << endl;
	//cout << scores;
	return image_origin;
	//return scores;

}

inline Mat computeVpScore(string filePath)
{
	Mat img = imread(filePath);
	return computeVpScore(img);
}

void visitVideo(string filePath)
{
	VideoCapture video(filePath);
	if (!video.isOpened())
	{
		cout << "failed to open the video" << endl;
	}
	size_t frames = video.get(CAP_PROP_FRAME_COUNT);
	size_t i = 0;
	Mat img, labeled;
	namedWindow("frame");
	while (i < frames)
	{
		video >> img;
		if (img.cols * img.rows > 0)
		{
			Mat labeled = computeVpScore(img);
			imshow("frame", labeled);
			waitKey(1);
		}
	}
}

void testFunc(int arr[][10])
{
	cout << "haha\n";
}

int main()
{
	int **x;
	//testFunc(x);
	//return 0;

	for (int j = 0; j < 200; j++)
	{
		for (int i = 0; i < 400; i++)
		{
			float t1 = sqrtf(pow(i - 200.0f, 2.0f) + pow(j, 2.0f));
			arcMatrix[j][i].arc = acos((200.0f - (float)i) / t1) / PI * 180.0f;
			arcMatrix[j][i].d = t1;
			//cout << j << ' ' << i << ' ' << arcMatrix[j][i].arc << ' ' << t1<<"   ";
		}
		//cout << endl;
	}
	int n_theta = 36;
	for (int t = 0; t < n_theta; t++)
	{
		float theta = PI*(float)t / (float)n_theta;
		Mat oddKernel(17, 17, CV_32F), evenKernel(17, 17, CV_32F);
		myGaborKernel(theta, oddKernels[t], evenKernels[t]);
	}
	int m = 200, n = 200;
	gabors = new int**[m];
	for_each(gabors, gabors + m, [n](int** &x){x = new int*[n]; });
	for_each(gabors, gabors + m, [n, n_theta](int** x){for_each(x, x + n, [&, n_theta](int* &y){ y = new int[n_theta]; }); });

	visit("C:\\images\\test", computeVpScore);
	//visitVideo("C:\\images\\video\\IMG_0307.m4v");

	//Release memory
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			delete[] gabors[i][j];
	for (int i = 0; i < m; i++)
		delete[] gabors[i];
	delete[] gabors;
	system("pause");
	return 0;
}