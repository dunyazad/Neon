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

	class Mesh : public ComponentBase
	{
	public:
		Mesh(const string& name);
		~Mesh();

		inline VertexArrayObject* GetVAO() { return vao; }

		VertexBufferObject<float>* GetVertexBuffer();
		VertexBufferObject<float>* GetNormalBuffer();
		VertexBufferObject<GLuint>* GetIndexBuffer();
		VertexBufferObject<float>* GetColorBuffer();
		VertexBufferObject<float>* GetUVBuffer();

		void AddVertex(float x, float y, float z);
		void GetVertex(int index, float& x, float& y, float& z);
		void SetVertex(int index, float x, float y, float z);
		void AddNormal(float x, float y, float z);
		void GetNormal(int index, float& x, float& y, float& z);
		void SetNormal(int index, float x, float y, float z); 
		void AddIndex(GLuint index);
		void AddColor(float r, float g, float b, float a);
		void AddUV(float u, float v);

		void Bind();
		void Unbind();

		inline GLenum GetDrawingMode() { return drawingMode; }
		inline void SetDrawingMode(GLenum mode) { drawingMode = mode; }

		void FromSTLFile(const string& filePath, float scaleX = 1.0f, float scaleY = 1.0f, float scaleZ = 1.0f);
		void RecalculateFaceNormal();

	private:
		VertexArrayObject* vao = nullptr;
		map<VertexBufferObjectBase::BufferType, VertexBufferObjectBase*> bufferObjects;
		GLenum drawingMode = GL_TRIANGLES;
	};
}
