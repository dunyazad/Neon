#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonComponent.h>

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
		RenderData();
		~RenderData();

		inline VertexArrayObject* GetVAO() { return vao; }

		VertexBufferObject<float>* GetVertexBuffer();
		VertexBufferObject<float>* GetNormalBuffer();
		VertexBufferObject<GLuint>* GetIndexBuffer();
		VertexBufferObject<float>* GetColorBuffer();
		VertexBufferObject<float>* GetUVBuffer();

		inline set<Shader*>& GetShaders() { return shaders; }
		inline vector<Texture*>& GetTextures() { return textures; }

		void AddVertex(float x, float y, float z);
		void AddNormal(float x, float y, float z);
		void AddIndex(GLuint index);
		void AddColor(float r, float g, float b, float a);
		void AddUV(float u, float v);

		inline void AddShader(Shader* shader) { shaders.insert(shader); }
		inline void AddTexture(Texture* texture) { textures.push_back(texture); }

		void Bind();
		void Unbind();

	private:
		VertexArrayObject* vao = nullptr;
		map<VertexBufferObjectBase::BufferType, VertexBufferObjectBase*> bufferObjects;
		set<Shader*> shaders;
		vector<Texture*> textures;
	};
}
