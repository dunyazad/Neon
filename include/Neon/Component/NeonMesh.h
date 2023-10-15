#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonURL.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class VertexArrayObject;
	class VertexBufferObjectBase;
	template <typename T>
	class VertexBufferObject;
	class Shader;
	class Texture;

	class Mesh : public ComponentBase
	{
	public:
		Mesh(const string& name);
		~Mesh();

		void Clear();

		inline VertexArrayObject* GetVAO() { return vao; }

		VertexBufferObject<glm::vec3>* GetVertexBuffer();
		VertexBufferObject<glm::vec3>* GetNormalBuffer();
		VertexBufferObject<GLuint>* GetIndexBuffer();
		VertexBufferObject<glm::vec4>* GetColorBuffer();
		VertexBufferObject<glm::vec2>* GetUVBuffer();

		size_t AddVertex(const glm::vec3& v);
		const glm::vec3& GetVertex(size_t index);
		void SetVertex(size_t index, const glm::vec3& v);

		size_t AddNormal(const glm::vec3& n);
		const glm::vec3& GetNormal(size_t index);
		void SetNormal(size_t index, const glm::vec3& n);

		size_t AddIndex(GLuint index);
		void GetIndex(size_t bufferIndex, GLuint& index);

		size_t AddColor(const glm::vec4& c);
		const glm::vec4& GetColor(size_t index);
		void SetColor(size_t index, const glm::vec4& c);

		size_t AddUV(const glm::vec2& uv);
		const glm::vec2& GetUV(size_t index);
		void SetUV(size_t index, const glm::vec2& uv);

		void Bind();
		void Unbind();

		void FromSTLFile(const URL& fileURL, float scaleX = 1.0f, float scaleY = 1.0f, float scaleZ = 1.0f);
		void RecalculateFaceNormal();
		void FillColor(const glm::vec4& color);

		bool Pick(const Ray& ray, glm::vec3& intersection, size_t& faceIndex);

		enum FillMode { Fill, Line, Point };

		inline GLenum GetDrawingMode() { return drawingMode; }
		inline void SetDrawingMode(GLenum mode) { drawingMode = mode; }

		inline FillMode GetFillMode() { return fillMode; }
		inline void SetFillMode(FillMode mode) { fillMode = mode; }
		inline void ToggleFillMode() { int f = fillMode; f++; fillMode = (Mesh::FillMode)(f % ((int)Mesh::FillMode::Point + 1)); }

		inline bool IsVisible() const { return visible; }
		inline void SetVisible(bool visible) { this->visible = visible; }

	private:
		bool visible = true;

		VertexArrayObject* vao = nullptr;
		map<VertexBufferObjectBase::BufferType, VertexBufferObjectBase*> bufferObjects;
		GLenum drawingMode = GL_TRIANGLES;
		FillMode fillMode = Fill;
	};
}
