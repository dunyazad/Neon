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
		//position.x = distance * sinf(glm::radians(azimuth)) * cosf(glm::radians(elevation));
		//position.y = distance * cosf(glm::radians(azimuth));
		//position.z = distance * sinf(glm::radians(azimuth)) * sinf(glm::radians(elevation));

		//cout << position << endl;
		//viewMatrix = glm::lookAt(position, centerPosition, glm::vec3(0, 1, 0));

		if (angleV > 90.0f) angleV = 90.0f;
		else if (angleV < -90.0f) angleV = -90.0f;

		auto rH = glm::angleAxis(glm::radians(angleH - 180), glm::vec3(0, 1, 0));
		auto rV = glm::angleAxis(glm::radians(angleV - 180), glm::vec3(1, 0, 0));
		auto eye = centerPosition + (rH * rV) * glm::vec3(0, 0, distance);
		position = eye;

		viewMatrix = glm::lookAt(eye, centerPosition, (rH * rV) * glm::vec3(0, -1, 0));
		projectionMatrix = glm::perspective(fovy, frameWidth / frameHeight, zNear, zFar);

		Transform::OnUpdate(now, timeDelta);
	}
}
