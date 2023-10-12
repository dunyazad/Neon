#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Camera;

	class CameraManipulator : public ComponentBase
	{
	public:
		CameraManipulator(const string& name, Camera* camera);
		virtual ~CameraManipulator();

		virtual void OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);
		virtual void OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods);
		virtual void OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos);
		virtual void OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset);

	protected:
		Camera* camera = nullptr;

		bool isShiftDown = false;

		bool isLButtonPressed = false;
		bool isRButtonPressed = false;
		bool isMButtonPressed = false;

		double lbuttonPressX = 0.0;
		double lbuttonPressY = 0.0;

		double rbuttonPressX = 0.0;
		double rbuttonPressY = 0.0;

		double mbuttonPressX = 0.0;
		double mbuttonPressY = 0.0;

		double lastCursorPosX = 0.0;
		double lastCursorPosY = 0.0;
	};
}