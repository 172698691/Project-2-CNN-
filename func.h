#pragma once
#ifndef FUNC_H
#define FUNC_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Tensor 
{
public:
	unsigned row;
	unsigned col;
	unsigned channel;
	float* data;
	Tensor()
	{
		row = 1;
		col = 1;
		channel = 1;
		data = new float[1]{ 0 };
	}
	Tensor(unsigned r, unsigned c, unsigned cha)
	{
		row = r;
		col = c;
		channel = cha;
		data = new float[r * c * cha]{ 0 };
	}
	~Tensor()
	{
		delete[] data;
	}
};

float* dealimage(Mat image, unsigned row, unsigned col);
void print(Tensor& t);
float dot1(float* v1, unsigned s1, unsigned e1, float* v2, unsigned s2, unsigned e2);
float dot2(float* v1, unsigned s1, unsigned e1, float* v2, unsigned s2, unsigned e2);
void conv(Tensor& in, Tensor& out, int layer);
void addzero(Tensor& t);
void relu(Tensor& t);
float max(float f1, float f2, float f3, float f4);
void max_pooling(Tensor& in, Tensor& out);
void full_connect(Tensor& in, Tensor& out);
float* softmax(Tensor& out);


#endif