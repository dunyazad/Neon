#include <Neon/NeonRenderData.h>
#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonTexture.h>

namespace Neon
{
	RenderData::RenderData()
	{
		vao = new VertexArrayObject();
	}

	RenderData::~RenderData()
	{
		for (auto& kvp : bufferObjects)
		{
			SAFE_DELETE(kvp.second);
		}

		SAFE_DELETE(vao);
	}

	VertexBufferObject<float>* RenderData::GetVertexBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::VERTEX_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::VERTEX_BUFFER, VertexBufferObjectBase::BufferType::VERTEX_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::VERTEX_BUFFER] = buffer;
		}
		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::VERTEX_BUFFER];
	}

	VertexBufferObject<float>* RenderData::GetNormalBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::NORMAL_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::NORMAL_BUFFER, VertexBufferObjectBase::BufferType::NORMAL_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::NORMAL_BUFFER] = buffer;
		}

		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::NORMAL_BUFFER];
	}

	VertexBufferObject<GLuint>* RenderData::GetIndexBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::INDEX_BUFFER))
		{
			auto buffer = new VertexBufferObject<GLuint>(VertexBufferObjectBase::BufferType::INDEX_BUFFER, VertexBufferObjectBase::BufferType::INDEX_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::INDEX_BUFFER] = buffer;
		}

		return (VertexBufferObject<GLuint>*)bufferObjects[VertexBufferObjectBase::BufferType::INDEX_BUFFER];
	}

	VertexBufferObject<float>* RenderData::GetColorBuffer() 
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::COLOR_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::COLOR_BUFFER, VertexBufferObjectBase::BufferType::COLOR_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::COLOR_BUFFER] = buffer;
		}

		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::COLOR_BUFFER];
	}

	VertexBufferObject<float>* RenderData::GetUVBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::UV_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::UV_BUFFER, VertexBufferObjectBase::BufferType::UV_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::UV_BUFFER] = buffer;
		}

		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::UV_BUFFER];
	}

	void RenderData::AddVertex(float x, float y, float z)
	{
		auto buffer = GetVertexBuffer();
		buffer->AddElement(x);
		buffer->AddElement(y);
		buffer->AddElement(z);
	}

	void RenderData::AddNormal(float x, float y, float z)
	{
		auto buffer = GetNormalBuffer();
		buffer->AddElement(x);
		buffer->AddElement(y);
		buffer->AddElement(z);
	}

	void RenderData::AddIndex(GLuint index)
	{
		auto buffer = GetIndexBuffer();
		buffer->AddElement(index);
	}

	void RenderData::AddColor(float r, float g, float b, float a)
	{
		auto buffer = GetColorBuffer();
		buffer->AddElement(r);
		buffer->AddElement(g);
		buffer->AddElement(b);
		buffer->AddElement(a);
	}
	
	void RenderData::AddUV(float u, float v)
	{
		auto buffer = GetUVBuffer();
		buffer->AddElement(u);
		buffer->AddElement(v);
	}

	void RenderData::Bind()
	{
		vao->Bind();

		for (auto& kvp : bufferObjects)
		{
			kvp.second->Bind();
			kvp.second->Upload();
		}

		for (size_t i = 0; i < textures.size(); i++)
		{
			textures[i]->Bind(GL_TEXTURE0 + i);
		}
	}

	void RenderData::Unbind()
	{
		vao->Unbind();

		for (auto& kvp : bufferObjects)
		{
			kvp.second->Unbind();
		}

		for (auto& texture : textures)
		{
			texture->Unbind();
		}
	}
}
