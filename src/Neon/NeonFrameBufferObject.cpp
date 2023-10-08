#include <Neon/NeonFrameBufferObject.h>
#include <Neon/NeonTexture.h>

namespace Neon
{
	int count = 0;

	FrameBufferObject::FrameBufferObject(const string& name, int width, int height)
		: width(width), height(height)
	{
		glGenFramebuffers(1, &fboID);
		glBindFramebuffer(GL_FRAMEBUFFER, fboID);

		if (targetTexture == nullptr)
		{
			stringstream ss;
			ss << "FrameBufferObject Texture " << count++;
			targetTexture = new Texture("name.targetTexture", width, height);
		}


		targetTexture->Bind();
		glGenerateMipmap(GL_TEXTURE_2D);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targetTexture->GetTextureID(), 0);

		glGenRenderbuffers(1, &depthBufferID);
		glBindRenderbuffer(GL_RENDERBUFFER, depthBufferID);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBufferID);

		//GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		//if (status == GL_FRAMEBUFFER_COMPLETE)
		//{
		//	printf("Framebuffer OK\n");
		//}
		//else
		//{
		//	printf("Framebuffer NOT OK\n");
		//}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	FrameBufferObject::FrameBufferObject(const string& name, Texture* texture)
		: targetTexture(texture)
	{
		if (targetTexture != nullptr)
		{
			width = targetTexture->GetWidth();
			height = targetTexture->GetHeight();
		}

		glGenFramebuffers(1, &fboID);
		glBindFramebuffer(GL_FRAMEBUFFER, fboID);

		if (targetTexture == nullptr)
		{
			stringstream ss;
			ss << "FrameBufferObject Texture " << count++;
			targetTexture = new Texture("name.targetTexture", width, height);
		}


		targetTexture->Bind();
		glGenerateMipmap(GL_TEXTURE_2D);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targetTexture->GetTextureID(), 0);

		glGenRenderbuffers(1, &depthBufferID);
		glBindRenderbuffer(GL_RENDERBUFFER, depthBufferID);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBufferID);

		//GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		//if (status == GL_FRAMEBUFFER_COMPLETE)
		//{
		//	printf("Framebuffer OK\n");
		//}
		//else
		//{
		//	printf("Framebuffer NOT OK\n");
		//}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	FrameBufferObject::~FrameBufferObject()
	{
		if (depthBufferID != -1)
		{
			glDeleteRenderbuffers(1, &depthBufferID);
		}

		if (fboID != -1)
		{
			glDeleteFramebuffers(1, &fboID);
		}
	}

	void FrameBufferObject::Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fboID);
		targetTexture->Bind();
	}

	void FrameBufferObject::Unbind()
	{
		targetTexture->Unbind();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void FrameBufferObject::Resize(int width, int height)
	{
		this->width = width;
		this->height = height;

		targetTexture->Resize(width, height);

		if (depthBufferID != -1)
		{
			glDeleteRenderbuffers(1, &depthBufferID);
		}

		if (fboID != -1)
		{
			glDeleteFramebuffers(1, &fboID);
		}
	}
}
