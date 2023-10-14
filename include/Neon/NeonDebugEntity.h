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

		void AddPoint(const glm::vec3& v, const glm::vec4& c = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		void AddLine(const glm::vec3& v0, const glm::vec3& v1, const glm::vec4& c0 = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), const glm::vec4& c1 = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		void AddTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec4& c0 = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), const glm::vec4& c1 = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), const glm::vec4& c2 = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

	protected:
		Mesh* mesh = nullptr;
		Shader* shader = nullptr;
	};
}
