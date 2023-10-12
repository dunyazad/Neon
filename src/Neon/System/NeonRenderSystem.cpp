#include <Neon/System/NeonRenderSystem.h>

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
	RenderSystem::RenderSystem(Scene* scene)
		: SystemBase(scene)
	{

	}

	RenderSystem::~RenderSystem()
	{

	}

	void RenderSystem::Frame(float now, float timeDelta)
	{
		auto camera = scene->GetMainCamera();
		camera->OnUpdate(now, timeDelta);

		auto light = scene->GetMainLight();
		light->OnUpdate(now, timeDelta);

		auto entities = scene->GetEntities();
		for (auto& kvp : entities)
		{
			auto entity = kvp.second;
			if (nullptr == entity)
				continue;

			auto shader = entity->GetComponent<Shader>(0);
			if (nullptr != shader)
			{
				shader->Use();
			}

			if (nullptr != camera)
			{
				if (nullptr != shader)
				{
					shader->SetUniformFloat4x4("projection", glm::value_ptr(camera->projectionMatrix));
					shader->SetUniformFloat4x4("view", glm::value_ptr(camera->viewMatrix));
				}
			}

			if (nullptr != light)
			{
				if (nullptr != shader)
				{
					if (nullptr != camera)
					{
						shader->SetUniformFloat3("cameraPosition", glm::value_ptr(camera->position));
					}
					shader->SetUniformFloat3("lightPosition", glm::value_ptr(light->position));
					shader->SetUniformFloat3("lightDirection", glm::value_ptr(light->direction));
					shader->SetUniformFloat3("lightColor", glm::value_ptr(light->color));
				}
			}

			auto transform = entity->GetComponent<Transform>(0);
			if (nullptr != transform)
			{
				if (nullptr != shader)
				{
					shader->SetUniformFloat4x4("model", glm::value_ptr(transform->absoluteTransform));
				}
			}
			else
			{
				glm::mat4 identity = glm::identity<glm::mat4>();

				if (nullptr != shader)
				{
					shader->SetUniformFloat4x4("model", glm::value_ptr(identity));
				}
			}

			auto texture = entity->GetComponent<Texture>(0);
			if (nullptr != texture)
			{
				texture->Bind();
			}

			auto mesh = entity->GetComponent<Mesh>(0);
			if (nullptr != mesh)
			{
				mesh->Bind();
				glDrawElements(mesh->GetDrawingMode(), (GLsizei)mesh->GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
			}
		}
	}
}
