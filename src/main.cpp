#include <iostream>

#include <Neon/Neon.h>

int main()
{
	Neon::Application app(1280, 1024);
	Neon::URL::ChangeDirectory("..");

	Neon::Image* imageB = nullptr;
	Neon::Image* imageC = nullptr;

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		auto scene = app.CreateScene("Scene/Main");

		auto debugLines = scene->CreateDebugEntity("DebugEntity/Lines");
		auto debugTriangles = scene->CreateDebugEntity("DebugEntity/Triangles");
		debugTriangles->SetKeyEventCallback([debugTriangles](const Neon::KeyEvent& event) {
			auto mesh = debugTriangles->GetComponent<Neon::Mesh>(0);
			if (GLFW_KEY_4 == event.key)
			{
				mesh->SetFillMode(Neon::Mesh::Fill);
			}
			else if (GLFW_KEY_5 == event.key)
			{
				mesh->SetFillMode(Neon::Mesh::Line);
			}
			else if (GLFW_KEY_6 == event.key)
			{
				mesh->SetFillMode(Neon::Mesh::Point);
			}
			});

		{
			auto entity = scene->CreateEntity("Entity/Main Camera");
			auto transform = scene->CreateComponent<Neon::Transform>("Transform/Main Camera");
			entity->AddComponent(transform);

			auto camera = scene->CreateComponent<Neon::Camera>("Camera/Main", 1280.0f, 1024.0f);
			entity->AddComponent(camera);
			camera->distance = 2.0f;
			camera->angleH = 30.0f;
			camera->angleV = 30.0f;
			camera->centerPosition = glm::vec3(0.0f, 0.0f, 0.0f);
			scene->SetMainCamera(camera);

			auto cameraManipulator = scene->CreateComponent<Neon::CameraManipulator>("CameraManipulator/Main", entity, camera);
			entity->AddComponent(cameraManipulator);
		}

		{
			auto entity = scene->CreateEntity("Entity/Main Light");

			auto light = scene->CreateComponent<Neon::Light>("Light/Main");
			entity->AddComponent(light);

			light->SetUpdateCallback([scene, light](float now, float timeDelta) {
				auto camera = scene->GetMainCamera();
				light->position = camera->position;
				light->direction = glm::normalize(camera->centerPosition - camera->position);
				});

			scene->SetMainLight(light);
		}

		{
			auto entity = scene->CreateEntity("Entity/Axes");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Axes");
			entity->AddComponent(mesh);

			mesh->SetDrawingMode(GL_LINES);
			mesh->AddVertex(0.0f, 0.0f, 0.0f);
			mesh->AddVertex(10.0f, 0.0f, 0.0f);
			mesh->AddVertex(0.0f, 0.0f, 0.0f);
			mesh->AddVertex(0.0f, 10.0f, 0.0f);
			mesh->AddVertex(0.0f, 0.0f, 0.0f);
			mesh->AddVertex(0.0f, 0.0f, 10.0f);

			mesh->AddColor(1.0f, 0.0f, 0.0f, 1.0f);
			mesh->AddColor(1.0f, 0.0f, 0.0f, 1.0f);
			mesh->AddColor(0.0f, 1.0f, 0.0f, 1.0f);
			mesh->AddColor(0.0f, 1.0f, 0.0f, 1.0f);
			mesh->AddColor(0.0f, 0.0f, 1.0f, 1.0f);
			mesh->AddColor(0.0f, 0.0f, 1.0f, 1.0f);

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);
			mesh->AddIndex(3);
			mesh->AddIndex(4);
			mesh->AddIndex(5);

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Color", Neon::URL::Resource("/shader/color.vs"), Neon::URL::Resource("/shader/color.fs"));
			entity->AddComponent(shader);
		}

		{
			auto t = Neon::Time("Mesh Loading");

			auto entity = scene->CreateEntity("Entity/Mesh");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Mesh");
			entity->AddComponent(mesh);
			mesh->FromSTLFile(Neon::URL::Resource("/stl/EiffelTower_fixed.stl"), 0.01f, 0.01f, 0.01f);
			auto nov = mesh->GetVertexBuffer()->Size() / 3;
			auto rotation = glm::angleAxis(-glm::half_pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
			for (int i = 0; i < nov; i++)
			{
				mesh->AddColor(1.0f, 1.0f, 1.0f, 1.0f);

				{
					float x, y, z;
					mesh->GetVertex(i, x, y, z);
					auto roatated = rotation * glm::vec3(x, y, z);
					mesh->SetVertex(i, roatated.x, roatated.y, roatated.z);
				}

				//{
				//	float x, y, z;
				//	mesh->GetNormal(i, x, y, z);
				//	auto roatated = rotation * glm::vec3(x, y, z);
				//	mesh->SetNormal(i, roatated.x, roatated.y, roatated.z);
				//}
			}

			mesh->RecalculateFaceNormal();

			auto noi = (int)mesh->GetIndexBuffer()->Size();
			for (int i = 0; i < noi / 3; i++)
			{
				auto i0 = mesh->GetIndexBuffer()->GetElement(i * 3 + 0);
				auto i1 = mesh->GetIndexBuffer()->GetElement(i * 3 + 1);
				auto i2 = mesh->GetIndexBuffer()->GetElement(i * 3 + 2);

				glm::vec3 v0, v1, v2;
				mesh->GetVertex(i0, v0.x, v0.y, v0.z);
				mesh->GetVertex(i1, v1.x, v1.y, v1.z);
				mesh->GetVertex(i2, v2.x, v2.y, v2.z);

				debugTriangles->AddTriangle(v0, v1, v2, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
			}

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			auto transform = scene->CreateComponent<Neon::Transform>("Transform/Mesh");
			entity->AddComponent(transform);
			transform->SetUpdateCallback([transform](float now, float timeDelta) {
				//transform->rotation = glm::angleAxis(glm::radians(now * 0.01f), glm::vec3(0.0f, 1.0f, 0.0f));
				});

			entity->SetKeyEventCallback([mesh](const Neon::KeyEvent& event) {
				if (GLFW_KEY_1 == event.key)
				{
					mesh->SetFillMode(Neon::Mesh::Fill);
				}
				else if (GLFW_KEY_2 == event.key)
				{
					mesh->SetFillMode(Neon::Mesh::Line);
				}
				else if (GLFW_KEY_3 == event.key)
				{
					mesh->SetFillMode(Neon::Mesh::Point);
				}
				});
		}
		});





	app.OnUpdate([&](float now, float timeDelta) {
		//auto t = Neon::Time("Update");

		//fbo->Bind();

		//glClearColor(0.9f, 0.7f, 0.5f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//shaderA->use();
		//triangle.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)triangle.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//shaderB->use();
		//owl.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)owl.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//lion.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)lion.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//fbo->Unbind();

		glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//shaderA->Use();
		//triangle.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)triangle.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//shaderB->Use();
		//owl.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)owl.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//lion.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)lion.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//frame.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)frame.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
		});






	app.OnTerminate([&]() {
		auto t = Neon::Time("Terminate");

		SAFE_DELETE(imageB);
		SAFE_DELETE(imageC);

		});

	app.Run();

	return 0;
}
