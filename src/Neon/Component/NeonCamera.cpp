#include <Neon/Component/NeonCamera.h>

namespace Neon
{
	Camera::Camera(const string& name, float frameWidth, float frameHeight)
		: Transform(name), frameWidth(frameWidth), frameHeight(frameHeight)
	{
		position = glm::vec3(0, 0, 3.f);
		centerPosition = glm::vec3(0, 0, 0.f);
		viewMatrix = glm::lookAt(position, centerPosition, glm::vec3(0, 1, 0));
		projectionMatrix = glm::perspective(fovy, frameWidth / frameHeight, zNear, zFar);
	}

	Camera::~Camera()
	{
	}

	void Camera::OnUpdate(float now, float timeDelta)
	{
		auto eye = centerPosition + rotation * glm::vec3(0, 0, distance);
		position = eye;

		viewMatrix = glm::lookAt(eye, centerPosition, rotation * glm::vec3(0, 1, 0));
		projectionMatrix = glm::perspective(fovy, frameWidth / frameHeight, zNear, zFar);

		Transform::OnUpdate(now, timeDelta);
	}

	Ray Camera::GetPickingRay(double xpos, double ypos)
	{
		GLint viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);
		float winX = (float)xpos;
		float winY = frameHeight - (float)ypos;

		auto u = winX / frameWidth - 0.5f;
		auto v = winY / frameHeight - 0.5f;

		auto pp = glm::unProject(
			glm::vec3(winX, winY, 1),
			glm::identity<glm::mat4>(),
			projectionMatrix * viewMatrix,
			glm::vec4(0, 0, frameWidth, frameHeight));

		auto rayOrigin = glm::vec3(glm::inverse(viewMatrix)[3]);
		auto rayDirection = glm::normalize(pp - rayOrigin);

		return { rayOrigin, rayDirection };
	}
}
