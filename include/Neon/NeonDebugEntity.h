#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonEntity.h>

namespace Neon
{
	class Mesh;
	class Shader;

	class DebugEntity : public Entity
	{
	public:
		DebugEntity(const string& name, Scene* scene);
		~DebugEntity();

		void Clear();

		void AddPoint(const glm::vec3& v, const glm::vec4& c = glm::white);
		void AddLine(const glm::vec3& v0, const glm::vec3& v1, const glm::vec4& c0 = glm::white, const glm::vec4& c1 = glm::white);
		void AddTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec4& c0 = glm::white, const glm::vec4& c1 = glm::white, const glm::vec4& c2 = glm::white);
		void AddBox(const glm::vec3& center, float xLength, float yLength, float zLength, const glm::vec4& color = glm::white);
		void AddSphere(const glm::vec3& center, float radius, int segments, const glm::vec4& color = glm::white);
		void AddCylinder(const glm::vec3& bottomCenter, float bottomRadius, const glm::vec3& topCenter, float topRadius, int segments, bool cap, const glm::vec4& bottomColor = glm::white, const glm::vec4& topColor = glm::white);
	protected:
		Mesh* mesh = nullptr;
		Shader* shader = nullptr;
	};
}
