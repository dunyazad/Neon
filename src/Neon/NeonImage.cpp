#include <Neon/NeonImage.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

//#define __STDC_LIB_EXT1__
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace Neon
{
	Image::Image(const URL& fileURL, bool verticalFlip)
		: fileURL(fileURL), verticalFlip(verticalFlip)
	{
		std::ifstream imageFile(fileURL.path);
		if (!imageFile) {
			std::cout << "Failed to open image file : " << fileURL.path << std::endl;
			exit(EXIT_FAILURE);
		}

		stbi_set_flip_vertically_on_load(verticalFlip);
		data = stbi_load(fileURL.path.c_str(), &width, &height, &nrChannels, bits);
	}

	Image::~Image()
	{
		if (data != nullptr) {
			stbi_image_free(data);
		}
	}

	void Image::Write(const URL& outputFileURL, bool verticalFlip)
	{
		stbi_flip_vertically_on_write(verticalFlip);
		stbi_write_png(outputFileURL.path.c_str(), width, height, nrChannels, data, width * nrChannels);
	}

	Image* Image::ResizeToPOT(Image* from)
	{
		Image* result = new Image(from->fileURL, from->verticalFlip);

		auto wpot = NextPowerOf2(from->width);
		auto hpot = NextPowerOf2(from->height);
		auto n = wpot > hpot ? wpot : hpot;

		auto newData = new unsigned char[n * n * from->nrChannels];
		memset(newData, 255, n * n * from->nrChannels);

		for (size_t h = 0; h < from->height; h++)
		{
			int targetIndex = int(h * n * from->nrChannels);
			int sourceIndex = int(h * from->width * from->nrChannels);
			memcpy(newData + targetIndex, from->data + sourceIndex, from->width * from->nrChannels);
		}

		if (result->data != nullptr) {
			delete result->data;
		}
		result->data = newData;
		result->width = n;
		result->height = n;
		result->nrChannels = from->nrChannels;

		result->resizedToPOT = true;
		result->potResizedRatioW = (float)n / (float)from->width;
		result->potResizedRatioH = (float)n / (float)from->height;

		return result;
	}
}
