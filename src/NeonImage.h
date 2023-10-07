#pragma once

#include "NeonCommon.h"

namespace Neon
{
	class Image
	{
	public:
		Image(const string& name, const string& filename, bool verticalFlip = true);
		~Image();

		void Initialize();

		void Write(const string& outputFilename, bool verticalFlip = true);

		static Image* ResizeToPOT(Image* from);

		inline string GetName() { return filename; }

		inline int GetWidth() { return width; }
		inline int GetHeight() { return height; }
		inline int GetChannels() { return nrChannels; }

		inline unsigned char* Data() { return data; }

	private:
		string filename;

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