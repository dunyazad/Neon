#include <Neon/Component/NeonCamera.h>

namespace Neon
{
	Camera::Camera(const string& name, float frameWidth, float frameHeight)
		: Transform(name), frameWidth(frameWidth), frameHeight(frameHeight)
	{
		position = glm::vec3(0, 0, 3.f);
		centerPosition = glm::vec3(0, 0, 0.f);
		viewMatrix = glm::lookAt(position, centerPosition, glm::vec3(0, 1, 0));
		projectionMatrix = glm::perspective(45.0f, frameHeight / frameWidth, zNear, zFar);
	}

	Camera::~Camera()
	{
	}

	void Camera::OnUpdate(float now, float timeDelta)
	{
		viewMatrix = glm::lookAt(position, centerPosition, glm::vec3(0, 1, 0));
		projectionMatrix = glm::perspective(45.0f, frameHeight / frameWidth, zNear, zFar);
	}
}
