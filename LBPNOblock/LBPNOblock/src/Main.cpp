#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <ctime>
#include "ImgBlocks.h"

using namespace cv;

// get the histogram
unsigned int hist[256] = { };
ImgBlocks* lbp = nullptr;


// Rotation to Right
uint8_t LBPValue0(Mat& block, int i, int j) {

	uint8_t lval = 0;
	uint8_t pxVal = block.at<uint8_t>(i, j);

	lval += (pxVal < block.at<uint8_t>(i - 1, j - 1)) ? 1 : 0;
	lval += (pxVal < block.at<uint8_t>(i - 1, j    )) ? 2 : 0;
	lval += (pxVal < block.at<uint8_t>(i - 1, j + 1)) ? 4 : 0;
	lval += (pxVal < block.at<uint8_t>(i    , j + 1)) ? 8 : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j + 1)) ? 16 : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j    )) ? 32 : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j - 1)) ? 64 : 0;
	lval += (pxVal < block.at<uint8_t>(i    , j - 1)) ? 128 : 0;

	return lval;

}


//  45 degrees
uint8_t LBPValue45(Mat& block, int i, int j) {

	uint8_t lval = 0;
	uint8_t pxVal = block.at<uint8_t>(i, j);

	lval += (pxVal < block.at<uint8_t>(i - 1, j - 1)) ? 1 : 0;
	lval += (pxVal < block.at<uint8_t>(i    , j - 2)) ? 2 : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j - 1)) ? 4 : 0;
	lval += (pxVal < block.at<uint8_t>(i + 2, j    )) ? 8 : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j + 1)) ? 16 : 0;
	lval += (pxVal < block.at<uint8_t>(i    , j + 2)) ? 32 : 0;
	lval += (pxVal < block.at<uint8_t>(i - 1, j + 1)) ? 64 : 0;
	lval += (pxVal < block.at<uint8_t>(i - 2, j    )) ? 128 : 0;

	return lval;

}


// << operator overload to print histogram
std::ostream& operator<<(std::ostream& stream, unsigned int* hist) {
	for (int i = 0; i < 256; i++) {
		stream << "Color Value : " << i << "\tFrequency : " << hist[i] << std::endl;
	}
	return stream;
}


Mat paddedImage(Mat &img, int blocksize) {

	int height = img.rows;
	int width = img.cols;

	height = int(ceil(height / (float)blocksize)) * blocksize;
	width = int(ceil(width / (float)blocksize)) * blocksize;

	Mat pimg = Mat::zeros(height+4, width+4, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			pimg.at<uint8_t>(i+2, j+2) = img.at<uint8_t>(i, j);
		}
	}

	return pimg;

}


void calcHist(Mat& image, unsigned int *histogram) {
	
	for (int i = 0; i < 256; i++)
		hist[i] = 0;

	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			hist[image.at<uint8_t>(i, j)]++;

}


// Draw Histogram
void histDisplay(unsigned int histogram[], const char* name)
{
	unsigned int hist[256];
	for (int i = 0; i < 256; i++)
	{
		hist[i] = histogram[i];
	}
	// draw the histograms
	int hist_w = 800; int hist_h = 600;
	int bin_w = cvRound((double)hist_w / 256);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

	// find the maximum intensity element from histogram
	int max = hist[0];
	for (int i = 1; i < 256; i++) {
		if (max < hist[i]) {
			max = hist[i];
		}
	}

	// normalize the histogram between 0 and histImage.rows
	for (int i = 0; i < 256; i++)
	{
		hist[i] = ((double)hist[i] / max) * histImage.rows;
	}


	// draw the intensity line for histogram
	for (int i = 0; i < 256; i++)
	{
		rectangle(histImage, Point(10 + bin_w * (i), hist_h), Point(10 + bin_w * (i) + bin_w, hist_h - hist[i]), Scalar(0, 0, 0));
	}

	// display histogram
	imshow(name, histImage);
}






// On mouse click display Histogram
static void onMouse(int event, int x, int y, int, void*)
{
	if (event != EVENT_LBUTTONDOWN)
		return;

	
	calcHist((lbp->getBlock(x/16, y/16)).b, hist);
	histDisplay(hist, "hist");
}


int main(int argc, char** argv)
{

	// to calculate execution time
	std::clock_t start;
	double duration;
	start = std::clock();
	
	// open image
	Mat image = imread("images/test2.jpg", IMREAD_GRAYSCALE);

	//padded image with 0 borders
	Mat padded_img = paddedImage(image, 16);

	// create lbp image
	Mat lbp_img0 = Mat::zeros(padded_img.rows, padded_img.cols, 0);
	Mat lbp_img45 = Mat::zeros(padded_img.rows, padded_img.cols, 0);

	// calculate and construct lbp image
	// we compare each pixel with his nighbors
	// we are ignoring LBPValues of the image border
	for (int i = 2; i < padded_img.rows - 2; i++) {
		for (int j = 2; j < padded_img.cols - 2; j++) {
			lbp_img0.at<uint8_t>(i, j) = LBPValue0(padded_img, i, j);
			lbp_img45.at<uint8_t>(i, j) = LBPValue45(padded_img, i, j);
		}
	}


	//Split Lbp
	ImgBlocks lbp_blocks(lbp_img0, 16);
	lbp = &lbp_blocks;

	// calculating time elapsed to get lbp image
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "Time Elapsed: " << duration << '\n';

	
	namedWindow("image", WINDOW_AUTOSIZE);
	namedWindow("lbp", WINDOW_AUTOSIZE);
	namedWindow("hist", WINDOW_AUTOSIZE);


	// display images
	imshow("image", image);
	setMouseCallback("image", onMouse, 0);

	imshow("lbp", lbp_img0);

	waitKey(0);
	destroyAllWindows();

	return 0;
}