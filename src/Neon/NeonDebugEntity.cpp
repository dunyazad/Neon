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

	void DebugEntity::AddLine(const glm::vec3& v0, const glm::vec3& v1, const glm::vec4& c0, const glm::vec4& c1)
	{
		GLuint index = (GLuint)mesh->GetVertexBuffer()->Size() / 3;
		mesh->AddVertex(v0.x, v0.y, v0.z);
		mesh->AddVertex(v1.x, v1.y, v1.z);
		mesh->AddColor(c0.r, c0.g, c0.b, c0.a);
		mesh->AddColor(c1.r, c1.g, c1.b, c1.a);
		mesh->AddIndex(index);
		mesh->AddIndex(index + 1);

		mesh->SetDrawingMode(GL_LINES);
		mesh->SetFillMode(Mesh::FillMode::Line);
	}

	void DebugEntity::AddTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec4& c0, const glm::vec4& c1, const glm::vec4& c2)
	{
		GLuint index = (GLuint)mesh->GetVertexBuffer()->Size() / 3;
		mesh->AddVertex(v0.x, v0.y, v0.z);
		mesh->AddVertex(v1.x, v1.y, v1.z);
		mesh->AddVertex(v2.x, v2.y, v2.z);
		mesh->AddColor(c0.r, c0.g, c0.b, c0.a);
		mesh->AddColor(c1.r, c1.g, c1.b, c1.a);
		mesh->AddColor(c2.r, c2.g, c2.b, c2.a);
		auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
		mesh->AddNormal(normal.x, normal.y, normal.z);
		mesh->AddNormal(normal.x, normal.y, normal.z);
		mesh->AddNormal(normal.x, normal.y, normal.z);
		mesh->AddIndex(index);
		mesh->AddIndex(index + 1);
		mesh->AddIndex(index + 2);
	}
}
