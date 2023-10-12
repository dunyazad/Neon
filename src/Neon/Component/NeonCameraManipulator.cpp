#include <Neon/Component/NeonCameraManipulator.h>

namespace Neon
{
	CameraManipulator::CameraManipulator(const string& name, Camera* camera)
		: ComponentBase(name), camera(camera)
	{
	}

	CameraManipulator::~CameraManipulator()
	{
	}

	void CameraManipulator::OnUpdate(float now, float timeDelta)
	{
	}
}
