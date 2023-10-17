#include <Neon/Component/NeonCameraManipulator.h>
#include <Neon/Component/NeonCamera.h>
#include <Neon/Component/NeonTransform.h>
#include <Neon/NeonEntity.h>

namespace Neon
{
	CameraManipulator::CameraManipulator(const string& name, Entity* cameraEntity, Camera* camera)
		: ComponentBase(name), cameraEntity(cameraEntity), camera(camera), transform(cameraEntity->GetComponent<Transform>(0))
	{
	}

	CameraManipulator::~CameraManipulator()
	{
	}

	void CameraManipulator::OnKeyEvent(const KeyEvent& event)
	{
		if (GLFW_KEY_LEFT_SHIFT == event.key || GLFW_KEY_RIGHT_SHIFT == event.key)
		{
			if (GLFW_PRESS == event.action)
			{
				isShiftDown = true;
			}
			else
			{
				isShiftDown = false;
			}
		}

		if (GLFW_KEY_SPACE == event.key && GLFW_RELEASE == event.action)
		{
			ResetRotation();
		}

		NeonObject::OnKeyEvent(event);
	}

	void CameraManipulator::OnMouseButtonEvent(const MouseButtonEvent& event)
	{
		if (event.button == GLFW_MOUSE_BUTTON_1) // Left Button
		{
			if (event.action == GLFW_PRESS)
			{
				isLButtonPressed = true;
				lbuttonPressX = lastCursorPosX;
				lbuttonPressY = lastCursorPosY;
			}
			else if (event.action == GLFW_RELEASE)
			{
				isLButtonPressed = false;
			}
		}
		else if (event.button == GLFW_MOUSE_BUTTON_2) // Right Button
		{
			if (event.action == GLFW_PRESS)
			{
				isRButtonPressed = true;

				rbuttonPressX = lastCursorPosX;
				rbuttonPressY = lastCursorPosY;
			}
			else if (event.action == GLFW_RELEASE)
			{
				isRButtonPressed = false;
			}
		}
		else if (event.button == GLFW_MOUSE_BUTTON_3) // Middle Button
		{
			if (event.action == GLFW_PRESS)
			{
				isMButtonPressed = true;

				mbuttonPressX = lastCursorPosX;
				mbuttonPressY = lastCursorPosY;
			}
			else if (event.action == GLFW_RELEASE)
			{
				isMButtonPressed = false;
			}
		}

		NeonObject::OnMouseButtonEvent(event);
	}

	void CameraManipulator::OnCursorPosEvent(const CursorPosEvent& event)
	{
		auto dx = event.xpos - lastCursorPosX;
		auto dy = event.ypos - lastCursorPosY;

		if (isRButtonPressed)
		{
			auto rH = glm::angleAxis(glm::radians(float(-dx * 0.25f)), glm::vec3(0, 1, 0));
			auto rV = glm::angleAxis(glm::radians(float(-dy * 0.25f)), glm::vec3(1, 0, 0));

			camera->rotation = camera->rotation * rH * rV;
		}

		if (isMButtonPressed)
		{
			auto xAxis = glm::vec3(glm::row(camera->viewMatrix, 0));
			auto yAxis = glm::vec3(glm::row(camera->viewMatrix, 1));

			transform->position += glm::normalize(xAxis) * camera->distance * 0.5f * (float)-dx * 0.001f;
			camera->centerPosition += glm::normalize(xAxis) * camera->distance * 0.5f * (float)-dx * 0.001f;
			transform->position += glm::normalize(yAxis) * camera->distance * 0.5f * (float)dy * 0.001f;
			camera->centerPosition += glm::normalize(yAxis) * camera->distance * 0.5f * (float)dy * 0.001f;
		}

		lastCursorPosX = event.xpos;
		lastCursorPosY = event.ypos;

		NeonObject::OnCursorPosEvent(event);
	}

	void CameraManipulator::OnScrollEvent(const ScrollEvent& event)
	{
		camera->fovy -= float(event.xoffset) * 0.01f;
		if (0.01f > camera->fovy) camera->fovy = 0.01f;
		if (89.99f < camera->fovy) camera->fovy = 89.99f;

		if (event.yoffset > 0)
		{
			camera->distance *= 0.9f;
		}
		else if (event.yoffset < 0)
		{
			camera->distance *= 1.1f;
		}

		NeonObject::OnScrollEvent(event);
	}

	void CameraManipulator::ResetRotation()
	{
		camera->rotation = glm::identity<glm::quat>();
	}
}
