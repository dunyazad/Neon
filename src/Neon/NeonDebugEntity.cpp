#include <Neon/NeonDebugEntity.h>
#include <Neon/NeonScene.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/Component/NeonComponent.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>

namespace Neon
{
	DebugEntity::DebugEntity(const string& name, Scene* scene)
		: Entity(name, scene)
	{
		mesh = scene->CreateComponent<Mesh>(name + "/mesh");
		this->AddComponent(mesh);
		shader = scene->CreateComponent<Neon::Shader>("Shader/debugging", Neon::URL::Resource("/shader/debugging.vs"), Neon::URL::Resource("/shader/debugging.fs"));
		this->AddComponent(shader);
	}

	DebugEntity::~DebugEntity()
	{
	}

	void DebugEntity::AddPoint(const glm::vec3& v, const glm::vec4& c)
	{
		auto index = mesh->AddVertex(v);
		mesh->AddColor(c);
		mesh->AddIndex((GLuint)index);

		mesh->SetDrawingMode(GL_POINTS);
		mesh->SetFillMode(Mesh::FillMode::Point);
	}

	void DebugEntity::AddLine(const glm::vec3& v0, const glm::vec3& v1, const glm::vec4& c0, const glm::vec4& c1)
	{
		GLuint index = (GLuint)mesh->GetVertexBuffer()->Size();
		mesh->AddVertex(v0);
		mesh->AddVertex(v1);
		mesh->AddColor(c0);
		mesh->AddColor(c1);
		mesh->AddIndex(index);
		mesh->AddIndex(index + 1);

		mesh->SetDrawingMode(GL_LINES);
		mesh->SetFillMode(Mesh::FillMode::Line);
	}

	void DebugEntity::AddTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec4& c0, const glm::vec4& c1, const glm::vec4& c2)
	{
		GLuint index = (GLuint)mesh->GetVertexBuffer()->Size();
		mesh->AddVertex(v0);
		mesh->AddVertex(v1);
		mesh->AddVertex(v2);
		mesh->AddColor(c0);
		mesh->AddColor(c1);
		mesh->AddColor(c2);
		auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
		mesh->AddNormal(normal);
		mesh->AddNormal(normal);
		mesh->AddNormal(normal);
		mesh->AddIndex(index);
		mesh->AddIndex(index + 1);
		mesh->AddIndex(index + 2);

		mesh->SetDrawingMode(GL_TRIANGLES);
		mesh->SetFillMode(Mesh::FillMode::Fill);
	}
}
