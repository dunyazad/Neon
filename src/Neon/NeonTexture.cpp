#include <Neon/NeonTexture.h>
#include <Neon/NeonImage.h>

namespace Neon
{
	Texture::Texture(const string& name, Image* image)
		: ComponentBase(name), image(image)
	{
		if (image != nullptr)
		{
			width = image->GetWidth();
			height = image->GetHeight();
			withAlpha = image->GetChannels() == 4;

			if (width != 0 && height != 0)
			{
				textureData = new unsigned char[width * height * image->GetChannels()];
				memset(textureData, 255, width * height * image->GetChannels());
				if (image->Data() != nullptr)
				{
					memcpy(textureData, image->Data(), width * height * image->GetChannels());
				}
			}
		}

		glGenTextures(1, &textureID);

		glBindTexture(textureTarget, textureID);

		glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLfloat borderColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		glTexParameterfv(textureTarget, GL_TEXTURE_BORDER_COLOR, borderColor);

		if (image == nullptr)
		{
			glTexImage2D(textureTarget, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}
		else
		{
			if (textureData != nullptr)
			{
				if (withAlpha)
				{
					glTexImage2D(textureTarget, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
				}
				else
				{
					glTexImage2D(textureTarget, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData);
				}
			}
		}
	}

	Texture::Texture(const string& name, int width, int height)
		: ComponentBase(name), width(width), height(height)
	{
		glGenTextures(1, &textureID);

		glBindTexture(textureTarget, textureID);

		glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLfloat borderColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		glTexParameterfv(textureTarget, GL_TEXTURE_BORDER_COLOR, borderColor);

		if (image == nullptr)
		{
			glTexImage2D(textureTarget, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}
		else
		{
			if (textureData != nullptr)
			{
				if (withAlpha)
				{
					glTexImage2D(textureTarget, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
				}
				else
				{
					glTexImage2D(textureTarget, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData);
				}
			}
		}
	}

	Texture::~Texture()
	{
		if (textureID != -1)
		{
			glDeleteTextures(1, &textureID);
		}

		if (textureData != nullptr) {
			delete textureData;
			textureData = nullptr;
		}
	}

	void Texture::Bind(GLenum textureSlot)
	{
		glActiveTexture(textureSlot);
		CheckGLError();

		glBindTexture(textureTarget, textureID);
		CheckGLError();
	}

	void Texture::Unbind()
	{
		glBindTexture(textureTarget, 0);
	}

	void Texture::Resize(int width, int height)
	{
		this->width = width;
		this->height = height;

		if (width != 0 && height != 0)
		{
			if (textureData != nullptr)
			{
				delete textureData;
				textureData = nullptr;
			}

			if (image != nullptr)
			{
				textureData = new unsigned char[width * height * image->GetChannels()];
				memset(textureData, 255, width * height * image->GetChannels());
				memcpy(textureData, image->Data(), width * height * image->GetChannels());
			}
		}

		glBindTexture(textureTarget, textureID);

		if (image == nullptr)
		{
			glTexImage2D(textureTarget, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}
		else
		{
			if (textureData != nullptr) {
				if (withAlpha) {
					glTexImage2D(textureTarget, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
				}
				else {
					glTexImage2D(textureTarget, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData);
				}
			}
		}
	}
}
