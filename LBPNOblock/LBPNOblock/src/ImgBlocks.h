#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <array>

class ImgBlocks
{

private:
	struct Block {
		cv::Mat b = cv::Mat::zeros(16, 16, 0);
	};

	Block* m_Blocks;
	int rowSize;

public:
	ImgBlocks(cv::Mat &image, int blockSize);
	~ImgBlocks();

	Block& getBlock(int i, int j) const;

};

