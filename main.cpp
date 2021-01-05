#include <opencv2/opencv.hpp>
#include<iostream>
#include <ctime>
#include <chrono>
#include<stdlib.h>
#include "func.h"

using namespace cv;
using namespace std;

int main()
{
    auto start = chrono::steady_clock::now();//��ʼ����֮ǰ��ʱ��
    Mat image = imread("10.jpg");
    Tensor pic(image.rows, image.cols, image.channels());
    delete[] pic.data;
    pic.data = dealimage(image, image.rows, image.cols);
    Tensor mid1;
    conv(pic, mid1, 1);
    relu(mid1);
    Tensor out1;
    max_pooling(mid1, out1);
    Tensor mid2;
    conv(out1, mid2, 2);
    relu(mid2);
    Tensor out2;
    max_pooling(mid2, out2);
    Tensor mid3;
    conv(out2, mid3, 3);
    relu(mid3);
    Tensor out;
    full_connect(mid3, out);
    float* p = softmax(out);
    cout.precision(2); cout << "score of background:" << p[0] << "\nscore of face:" << p[1] << endl;
    auto end = std::chrono::steady_clock::now();//������ɺ��ʱ��
    cout << "time using by calculate is " << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;//����ʱ�������������ʱ��

    return 0;
}