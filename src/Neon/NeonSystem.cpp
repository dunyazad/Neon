#include <Neon/NeonSystem.h>
#include <Neon/Neon.h>

namespace Neon
{
	SystemBase::SystemBase(Application* application)
		: application(application)
	{
	}

	SystemBase::~SystemBase()
	{
	}

	TransformUpdateSystem::TransformUpdateSystem(Application* application)
		: SystemBase(application)
	{
	}

	TransformUpdateSystem::~TransformUpdateSystem()
	{
	}

	void TransformUpdateSystem::Frame(float now, float timeDelta)
	{
		auto components = application->GetComponents<Transform>();
		for (auto& component : components)
		{
			((Transform*)component)->OnUpdate(now, timeDelta);
		}
	}


	RenderSystem::RenderSystem(Application* application)
		: SystemBase(application)
	{

	}

	RenderSystem::~RenderSystem()
	{

	}

	void RenderSystem::Frame(float now, float timeDelta)
	{
		auto entities = application->GetEntities();
		for (auto& kvp : entities)
		{
			auto entity = kvp.second;

			auto shader = entity->GetComponent<Shader>(0);
			if (nullptr != shader)
			{
				shader->Use();
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

			auto renderData = entity->GetComponent<Mesh>(0);
			if (nullptr != renderData)
			{
				renderData->Bind();
				glDrawElements(GL_TRIANGLES, (GLsizei)renderData->GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
			}
		}
	}
}
