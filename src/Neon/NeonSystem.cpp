#include <Neon/NeonSystem.h>
#include <Neon/NeonEntity.h>
#include <Neon/NeonScene.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/Component/NeonCamera.h>
#include <Neon/Component/NeonLight.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>
#include <Neon/Component/NeonTexture.h>
#include <Neon/Component/NeonTransform.h>

namespace Neon
{
	SystemBase::SystemBase(Scene* scene)
		: scene(scene)
	{
	}

	SystemBase::~SystemBase()
	{
	}

	TransformUpdateSystem::TransformUpdateSystem(Scene* scene)
		: SystemBase(scene)
	{
	}

	TransformUpdateSystem::~TransformUpdateSystem()
	{
	}

	void TransformUpdateSystem::Frame(float now, float timeDelta)
	{
		auto components = scene->GetComponents<Transform>();
		for (auto& component : components)
		{
			((Transform*)component)->OnUpdate(now, timeDelta);
		}
	}


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
					//auto lightDir = glm::vec3(0.0f, 0.0f, 10.0f);
					shader->SetUniformFloat3("lightPosition", glm::value_ptr(light->position));
					//shader->SetUniformFloat3("lightPos", glm::value_ptr(lightPos));
					shader->SetUniformFloat3("lightDirection", glm::value_ptr(light->direction));
					//shader->SetUniformFloat3("lightDirection", glm::value_ptr(lightDir));
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

	EventSystem::EventSystem(Scene* scene)
		: SystemBase(scene)
	{
	}

	EventSystem::~EventSystem()
	{
	}

	void EventSystem::Event()
	{

	}
}