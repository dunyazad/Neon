#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Texture;

	class FrameBufferObject
	{
	public:
		void Bind();
		void Unbind();

		void Resize(int width, int height);

		inline unsigned int GetFBOID() { return fboID; }
		inline Texture* GetTargetTexture() { return targetTexture; }

		inline int GetWidth() { return width; }
		inline int GetHeight() { return height; }

	protected:
		FrameBufferObject(const string& name, int width, int height);
		FrameBufferObject(const string& name, Texture* texture);
		~FrameBufferObject();

		GLuint fboID = -1;
		GLuint depthBufferID = -1;
		int width = 512;
		int height = 512;
		Texture* targetTexture = nullptr;
	};
}