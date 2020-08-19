#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <ctime>

using namespace cv;

class block {

private:
	mutable std::vector<Mat> row;

public:

	block(int width) {

		for (int i = 0; i < width; i++)
			row.push_back(Mat::zeros(18, 18, 0));

	}

	~block() = delete;

	
	Mat &getBlock(int i) const {
		return row[i];
	}

};

void copyRegion(const Mat &src, Mat &dst, int off, int offx, int offy) {

	for (int i = off; i < src.rows - off; i++)
		for (int j = off; j < src.cols - off; j++)
			dst.at<uint8_t>(i+offy, j+offx) = src.at<uint8_t>(i, j);

}

uint8_t LBPValue(Mat &block, int i, int j) {

	uint8_t lval = 0;
	uint8_t pxVal = block.at<uint8_t>(i, j);

	lval += (pxVal < block.at<uint8_t>(i - 1, j + 1)) ? 1   : 0;
	lval += (pxVal < block.at<uint8_t>(i    , j + 1)) ? 2   : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j + 1)) ? 4   : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j    )) ? 8   : 0;
	lval += (pxVal < block.at<uint8_t>(i + 1, j - 1)) ? 16  : 0;
	lval += (pxVal < block.at<uint8_t>(i    , j - 1)) ? 32  : 0;
	lval += (pxVal < block.at<uint8_t>(i - 1, j - 1)) ? 64  : 0;
	lval += (pxVal < block.at<uint8_t>(i - 1, j    )) ? 128 : 0;

	return lval;

}


std::ostream& operator<<(std::ostream& stream, unsigned int* hist) {
	for (int i = 0; i < 256; i++) {
		stream << "Color Value : " << i << "\tFrequency : " << hist[i] << std::endl;
	}
	return stream;
}

int main(int argc, char** argv)
{
	std::clock_t start;
	double duration;

	start = std::clock();
	
	Mat image = imread("images/testimage.png", IMREAD_GRAYSCALE);

	float blockSize = 16;

	size_t Xs = ceil(image.cols / blockSize);
	size_t Ys = ceil(image.rows / blockSize);

	int pheight = Ys * blockSize + 2;
	int pwidth = Xs * blockSize + 2;

	Mat padded_img = Mat::zeros(pheight, pwidth, 0);
	copyRegion(image, padded_img, 0, 1, 1);

	std::vector<block*> blocks;

	// devide into blocks 16x16 +2 for borders
	for (int i = 1; i < pheight - 1; i += blockSize) {

		blocks.emplace_back(new block(Xs));
		int counter = 0;
		for (int j = 1; j < pwidth - 1; j += blockSize) {

			for (int k = 0; k < 18; k++)
				for (int z = 0; z < 18; z++)
					blocks.back()->getBlock(counter).at<uint8_t>(k, z) = padded_img.at<uint8_t>(k + i - 1, z + j - 1);

			counter++;

		}
	}

	// LBP + Image Reconstruction
	Mat lbp_img = Mat::zeros(Size(pwidth - 2, pheight - 2), 0);
	Mat lbpblock = Mat::zeros(16, 16, 0);
	for (int i = 0; i < Ys; i++) {
		for (int j = 0; j < Xs; j++) {

			for (int k = 1; k < 17; k++)
				for (int z = 1; z < 17; z++)
					lbpblock.at<uint8_t>(k - 1, z - 1) = LBPValue(blocks[i]->getBlock(j), k, z);

			copyRegion(lbpblock, lbp_img, 0, j * blockSize, i * blockSize);
		}
	}
	
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "Time Elapsed: " << duration << '\n';

	unsigned int hist[256] = { };

	for (int i = 0; i < lbp_img.rows; i++)
		for (int j = 0; j < lbp_img.cols; j++)
			hist[lbp_img.at<uint8_t>(i, j)]++;
	
	std::cout << hist;


	namedWindow("test", WINDOW_AUTOSIZE);
	namedWindow("lbp", WINDOW_AUTOSIZE);

	imshow("test", padded_img);
	imshow("lbp", lbp_img);

	waitKey(0);

	return 0;
}