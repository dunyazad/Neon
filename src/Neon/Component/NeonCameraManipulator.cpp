#include <Neon/Component/NeonCameraManipulator.h>
#include <Neon/Component/NeonCamera.h>

namespace Neon
{
	CameraManipulator::CameraManipulator(const string& name, Camera* camera)
		: ComponentBase(name), camera(camera)
	{
	}

	CameraManipulator::~CameraManipulator()
	{
	}

	void CameraManipulator::OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
		{
			if (action == GLFW_PRESS)
			{
				isShiftDown = true;
			}
			else
			{
				isShiftDown = false;
			}
		}

		NeonObject::OnKeyEvent(window, key, scancode, action, mods);
	}

	void CameraManipulator::OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
	{
		if (button == GLFW_MOUSE_BUTTON_1) // Left Button
		{
			if (action == GLFW_PRESS)
			{
				isLButtonPressed = true;
				lbuttonPressX = lastCursorPosX;
				lbuttonPressY = lastCursorPosY;
			}
			else if (action == GLFW_RELEASE)
			{
				isLButtonPressed = false;
			}
		}
		else if (button == GLFW_MOUSE_BUTTON_2) // Right Button
		{
			if (action == GLFW_PRESS)
			{
				isRButtonPressed = true;

				rbuttonPressX = lastCursorPosX;
				rbuttonPressY = lastCursorPosY;
			}
			else if (action == GLFW_RELEASE)
			{
				isRButtonPressed = false;
			}
		}
		else if (button == GLFW_MOUSE_BUTTON_3) // Middle Button
		{
			if (action == GLFW_PRESS)
			{
				isMButtonPressed = true;

				mbuttonPressX = lastCursorPosX;
				mbuttonPressY = lastCursorPosY;
			}
			else if (action == GLFW_RELEASE)
			{
				isMButtonPressed = false;
			}
		}

		NeonObject::OnMouseButtonEvent(window, button, action, mods);
	}

	void CameraManipulator::OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos)
	{
		auto dx = xpos - lastCursorPosX;
		auto dy = ypos - lastCursorPosY;

		if (isRButtonPressed)
		{
			camera->angleH -= dx;
			camera->angleV += dy;
		}

		if (isMButtonPressed)
		{
			auto xAxis = glm::vec3(glm::row(camera->viewMatrix, 0));
			auto yAxis = glm::vec3(glm::row(camera->viewMatrix, 1));

			camera->position += glm::normalize(xAxis) * (float)-dx * 0.001f;
			camera->centerPosition += glm::normalize(xAxis) * (float)-dx * 0.001f;
			camera->position += glm::normalize(yAxis) * (float)dy * 0.001f;
			camera->centerPosition += glm::normalize(yAxis) * (float)dy * 0.001f;
		}

		lastCursorPosX = xpos;
		lastCursorPosY = ypos;

		NeonObject::OnCursorPosEvent(window, xpos, ypos);
	}

	void CameraManipulator::OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset)
	{
		camera->fovy -= float(xoffset) * 0.01f;
		if (0.01f > camera->fovy) camera->fovy = 0.01f;
		if (89.99f < camera->fovy) camera->fovy = 89.99f;

		if (yoffset > 0)
		{
			camera->distance *= 0.9f;
		}
		else
		{
			camera->distance *= 1.1f;
		}

		NeonObject::OnScrollEvent(window, xoffset, yoffset);
	}
}
