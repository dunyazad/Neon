#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonComponent.h>

namespace Neon
{
	class Image;

	class Texture : public ComponentBase
	{
	public:
		Texture(const string& name, Image* image);
		Texture(const string& name, int width, int height);
		~Texture();

		void Bind(GLenum textureSlot = GL_TEXTURE0);
		void Unbind();

		virtual void Resize(int width, int height);

		inline Image* GetImage() const { return image; }

		inline GLenum GetTarget() { return textureTarget; }
		inline GLuint GetTextureID() { return textureID; }

		inline GLsizei GetWidth() { return width; }
		inline GLsizei GetHeight() { return height; }

		inline bool HasAlpha() { return withAlpha; }

	protected:
		Image* image = nullptr;

		bool withAlpha = true;
		GLuint textureID = -1;
		GLenum textureTarget = GL_TEXTURE_2D; // GL_TEXTURE_2D, GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_CUBE_MAP
		GLenum format = GL_RGBA;
		GLsizei width = 0;
		GLsizei height = 0;
		GLenum dataType = GL_UNSIGNED_BYTE;
		unsigned char* textureData = nullptr;

	public:
		friend class HeGraphics;
	};
}