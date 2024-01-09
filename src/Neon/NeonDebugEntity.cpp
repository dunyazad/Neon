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

	void DebugEntity::Clear()
	{
		mesh->Clear();
	}

	void DebugEntity::AddMesh(Mesh* mesh)
	{
		for (auto& e : mesh->GetVertexBuffer()->GetElements())
		{
			this->mesh->AddVertex(e);
		}

		for (auto& e : mesh->GetNormalBuffer()->GetElements())
		{
			this->mesh->AddNormal(e);
		}

		for (auto& e : mesh->GetColorBuffer()->GetElements())
		{
			this->mesh->AddColor(e);
		}

		for (auto& e : mesh->GetUVBuffer()->GetElements())
		{
			this->mesh->AddUV(e);
		}

		for (auto& e : mesh->GetIndexBuffer()->GetElements())
		{
			this->mesh->AddIndex(e);
		}

		this->mesh->SetFillMode(mesh->GetFillMode());
		this->mesh->SetDrawingMode(mesh->GetDrawingMode());
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

	void DebugEntity::AddBox(const glm::vec3& center, float xLength, float yLength, float zLength, const glm::vec4& color)
	{
		float halfX = xLength * 0.5f;
		float halfY = yLength * 0.5f;
		float halfZ = zLength * 0.5f;

		float nx = center.x - halfX;
		float ny = center.y - halfY;
		float nz = center.z - halfZ;

		float px = center.x + halfX;
		float py = center.y + halfY;
		float pz = center.z + halfZ;

		AddTriangle(glm::vec3(nx, ny, nz), glm::vec3(nx, py, nz), glm::vec3(px, py, nz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, nz), glm::vec3(px, py, nz), glm::vec3(px, ny, nz), color, color, color);

		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(px, ny, pz), glm::vec3(px, py, pz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(px, py, pz), glm::vec3(nx, py, pz), color, color, color);

		AddTriangle(glm::vec3(nx, py, pz), glm::vec3(px, py, pz), glm::vec3(px, py, nz), color, color, color);
		AddTriangle(glm::vec3(nx, py, pz), glm::vec3(px, py, nz), glm::vec3(nx, py, nz), color, color, color);

		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(px, ny, nz), glm::vec3(px, ny, pz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(nx, ny, nz), glm::vec3(px, ny, nz), color, color, color);

		AddTriangle(glm::vec3(px, ny, pz), glm::vec3(px, ny, nz), glm::vec3(px, py, nz), color, color, color);
		AddTriangle(glm::vec3(px, ny, pz), glm::vec3(px, py, nz), glm::vec3(px, py, pz), color, color, color);

		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(nx, py, pz), glm::vec3(nx, py, nz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(nx, py, nz), glm::vec3(nx, ny, nz), color, color, color);
	}

	void DebugEntity::AddAABB(const AABB& aabb, const glm::vec4& color)
	{
		float halfX = aabb.GetXLength() * 0.5f;
		float halfY = aabb.GetYLength() * 0.5f;
		float halfZ = aabb.GetZLength() * 0.5f;

		float nx = aabb.GetCenter().x - halfX;
		float ny = aabb.GetCenter().y - halfY;
		float nz = aabb.GetCenter().z - halfZ;

		float px = aabb.GetCenter().x + halfX;
		float py = aabb.GetCenter().y + halfY;
		float pz = aabb.GetCenter().z + halfZ;

		AddTriangle(glm::vec3(nx, ny, nz), glm::vec3(nx, py, nz), glm::vec3(px, py, nz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, nz), glm::vec3(px, py, nz), glm::vec3(px, ny, nz), color, color, color);

		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(px, ny, pz), glm::vec3(px, py, pz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(px, py, pz), glm::vec3(nx, py, pz), color, color, color);

		AddTriangle(glm::vec3(nx, py, pz), glm::vec3(px, py, pz), glm::vec3(px, py, nz), color, color, color);
		AddTriangle(glm::vec3(nx, py, pz), glm::vec3(px, py, nz), glm::vec3(nx, py, nz), color, color, color);

		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(px, ny, nz), glm::vec3(px, ny, pz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(nx, ny, nz), glm::vec3(px, ny, nz), color, color, color);

		AddTriangle(glm::vec3(px, ny, pz), glm::vec3(px, ny, nz), glm::vec3(px, py, nz), color, color, color);
		AddTriangle(glm::vec3(px, ny, pz), glm::vec3(px, py, nz), glm::vec3(px, py, pz), color, color, color);

		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(nx, py, pz), glm::vec3(nx, py, nz), color, color, color);
		AddTriangle(glm::vec3(nx, ny, pz), glm::vec3(nx, py, nz), glm::vec3(nx, ny, nz), color, color, color);
	}

	void DebugEntity::AddSphere(const glm::vec3& center, float radius, int segments, const glm::vec4& color)
	{
		auto offset = mesh->GetVertexBuffer()->Size();

		for (int i = 0; i <= segments; ++i) {
			float phi = PI * float(i) / (float)segments;
			for (int j = 0; j <= segments; ++j) {
				float theta = PI * 2 * float(j) / (float)segments;
				float x = sinf(phi) * cosf(theta);
				float y = cosf(phi);
				float z = sinf(phi) * sinf(theta);

				auto v = glm::vec3{ x * radius, y * radius,  z * radius };
				mesh->AddVertex(v + center);
				mesh->AddColor(color);
				mesh->AddNormal(normalize(v));
			}
		}

		for (int i = 0; i < segments; ++i) {
			for (int j = 0; j < segments; ++j) {
				int current = i * (segments + 1) + j;
				int next = current + segments + 1;

				mesh->AddIndex((GLuint)offset + current);
				mesh->AddIndex((GLuint)offset + current + 1);
				mesh->AddIndex((GLuint)offset + next);

				mesh->AddIndex((GLuint)offset + next);
				mesh->AddIndex((GLuint)offset + current + 1);
				mesh->AddIndex((GLuint)offset + next + 1);
			}
		}
	}

	void AddCylinder(const glm::vec3& bottomCenter, float bottomRadius, const glm::vec3& topCenter, float topRadius, int segments, bool cap, const glm::vec4& bottomColor, const glm::vec4& topColor)
	{

	}
}
