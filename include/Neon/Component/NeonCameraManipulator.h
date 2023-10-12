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

		virtual void OnUpdate(float now, float timeDelta);

	protected:
		Camera* camera = nullptr;
	};
}