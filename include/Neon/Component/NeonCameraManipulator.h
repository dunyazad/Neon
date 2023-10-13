#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Entity;
	class Camera;
	class Transform;

	class CameraManipulator : public ComponentBase
	{
	public:
		CameraManipulator(const string& name, Entity* cameraEntity, Camera* camera);
		virtual ~CameraManipulator();

		virtual void OnKeyEvent(const KeyEvent& event);
		virtual void OnMouseButtonEvent(const MouseButtonEvent& event);
		virtual void OnCursorPosEvent(const CursorPosEvent& event);
		virtual void OnScrollEvent(const ScrollEvent& event);

	protected:
		Entity* cameraEntity = nullptr;
		Camera* camera = nullptr;
		Transform* transform = nullptr;

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