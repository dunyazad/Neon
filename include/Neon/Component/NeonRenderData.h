#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class VertexArrayObject;
	class VertexBufferObjectBase;
	template <class T>
	class VertexBufferObject;
	class Shader;
	class Texture;

	class RenderData : public ComponentBase
	{
	public:
		RenderData(const string& name);
		~RenderData();

		inline VertexArrayObject* GetVAO() { return vao; }

		VertexBufferObject<float>* GetVertexBuffer();
		VertexBufferObject<float>* GetNormalBuffer();
		VertexBufferObject<GLuint>* GetIndexBuffer();
		VertexBufferObject<float>* GetColorBuffer();
		VertexBufferObject<float>* GetUVBuffer();

		void AddVertex(float x, float y, float z);
		void AddNormal(float x, float y, float z);
		void AddIndex(GLuint index);
		void AddColor(float r, float g, float b, float a);
		void AddUV(float u, float v);

		void Bind();
		void Unbind();

	private:
		VertexArrayObject* vao = nullptr;
		map<VertexBufferObjectBase::BufferType, VertexBufferObjectBase*> bufferObjects;
	};
}
