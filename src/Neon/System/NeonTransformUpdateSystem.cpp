#include <Neon/System/NeonTransformUpdateSystem.h>

#include <Neon/NeonEntity.h>
#include <Neon/NeonScene.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonWindow.h>

#include <Neon/Component/NeonCamera.h>
#include <Neon/Component/NeonLight.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>
#include <Neon/Component/NeonTexture.h>
#include <Neon/Component/NeonTransform.h>

namespace Neon
{
	TransformUpdateSystem::TransformUpdateSystem(Scene* scene)
		: SystemBase(scene)
	{
	}

	TransformUpdateSystem::~TransformUpdateSystem()
	{
	}

	void TransformUpdateSystem::Frame(double now, double timeDelta)
	{
		auto components = scene->GetComponents<Transform>();
		for (auto& component : components)
		{
			((Transform*)component)->OnUpdate(now, timeDelta);
		}
	}
}
