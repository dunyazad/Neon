#include "NeonImage.h"

namespace Neon
{
	Image::Image(const string& name, const string& filename, bool verticalFlip)
		: filename(filename), verticalFlip(verticalFlip)
	{
	}

	Image::~Image()
	{
		if (data != nullptr) {
			stbi_image_free(data);
		}
	}

	void Image::Initialize()
	{
		stbi_set_flip_vertically_on_load(verticalFlip);
		data = stbi_load(filename.c_str(), &width, &height, &nrChannels, bits);
	}

	void Image::Write(const string& outputFilename, bool verticalFlip)
	{
		stbi_flip_vertically_on_write(verticalFlip);
		stbi_write_png(outputFilename.c_str(), width, height, nrChannels, data, width * nrChannels);
	}

	Image* Image::ResizeToPOT(Image* from)
	{
		Image* result = new Image(from->GetName() + "_resized", from->filename, from->verticalFlip);

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
