#include "ImgBlocks.h"


ImgBlocks::ImgBlocks(cv::Mat& image, int blockSize) {

	rowSize = image.rows / blockSize;

	//Allocate
	m_Blocks = new ImgBlocks::Block[rowSize * (image.cols / blockSize)];

	int k = 0;

	for (int i = 0; i < image.rows - blockSize; i += blockSize)
		for (int j = 0; j < image.cols - blockSize; j += blockSize) {

			for (int ii = 0; ii < blockSize; ii++)
				for (int jj = 0; jj < blockSize; jj++)
					m_Blocks[k].b.at<uint8_t>(ii, jj) = image.at<uint8_t>(i+ii, j+jj);
			
			k++;
		}

}

ImgBlocks::~ImgBlocks() {

	delete[] m_Blocks;

}

ImgBlocks::Block& ImgBlocks::getBlock(int i, int j) const {
	return m_Blocks[j + i * rowSize];
}
