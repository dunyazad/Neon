#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>
#include <Neon/Component/NeonTransform.h>

namespace Neon
{
	class Camera : public Transform
	{
	public:
		Camera(const string& name, float frameWidth, float frameHeight);
		virtual ~Camera();

		virtual void OnUpdate(float now, float timeDelta);

		float frameWidth = 1024.0f;
		float frameHeight = 768.0f;
		float zNear = 0.01f;
		float zFar = 3000.0f;

		glm::vec3 centerPosition = glm::zero<glm::vec3>();
		float distance = 5.0f;
		float azimuth = 0.0f;
		float elevation = 0.0f;

		glm::mat4 viewMatrix = glm::identity<glm::mat4>();
		glm::mat4 projectionMatrix = glm::identity<glm::mat4>();

	private:
	};
}