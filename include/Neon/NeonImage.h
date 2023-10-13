#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonURL.h>

namespace Neon
{
	class Image
	{
	public:
		Image(const URL& fileURL, bool verticalFlip = true);
		~Image();

		void Write(const URL& outputFileURL, bool verticalFlip = true);

		static Image* ResizeToPOT(Image* from);

		inline int GetWidth() { return width; }
		inline int GetHeight() { return height; }
		inline int GetChannels() { return nrChannels; }

		inline unsigned char* Data() { return data; }

	private:
		URL fileURL;

		int width = 0;
		int height = 0;
		int nrChannels = 0;
		int bits = 0;

		unsigned char* data = nullptr;

		bool verticalFlip = true;

		bool resizedToPOT = false;
		float potResizedRatioW = 1.0f;
		float potResizedRatioH = 1.0f;
	};

}