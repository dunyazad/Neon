#include <iostream>

#include <Neon/Neon.h>

int main()
{
	Neon::Application app(1280, 1024);
	app.SetResourceRoot(filesystem::current_path().string() + "/../res");

	Neon::Image* imageB = nullptr;
	Neon::Image* imageC = nullptr;

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		auto scene = app.CreateScene("Scene/Main");

		{
			auto entity = scene->CreateEntity("Entity/Main Camera");
			auto camera = scene->CreateComponent<Neon::Camera>("Camera/Main", 1280.0f, 1024.0f);
			entity->AddComponent(camera);
			camera->AddUpdateCallback([camera](float now, float timeDelta) {
				//camera->distance = sinf(now * 0.001f);
				camera->azimuth += timeDelta * 0.05f;
				//camera->elevation += timeDelta * 0.01f;
				});
			camera->distance = 2.0f;
			camera->azimuth = 45.0f;
			camera->elevation = -45.0f;
			camera->centerPosition = glm::vec3(0.0f, 0.0f, 0.0f);
			scene->SetMainCamera(camera);

			scene->SubscribeKeyEvent(entity);
			entity->SetKeyEventCallback([camera](GLFWwindow* window, int key, int scancode, int action, int mods) {
				if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
				{
					camera->azimuth -= 1.0f;
				}
				else if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT))
				{
					camera->azimuth += 1.0f;
				}
				});
		}

		{
			auto entity = scene->CreateEntity("Entity/Main Light");
			auto light = scene->CreateComponent<Neon::Light>("Light/Main");
			entity->AddComponent(light);

			light->AddUpdateCallback([scene, light](float now, float timeDelta) {
				auto camera = scene->GetMainCamera();
				light->position = camera->position;
				light->direction = glm::normalize(camera->centerPosition - camera->position);
				});

			//light->position = glm::vec3(0.0f, 0.0f, 10.0f);
			//light->direction = glm::vec3(0.0f, 0.0f, -1.0f);
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

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Color", (app.GetResourceRoot() + "/shader/color.vs").c_str(), (app.GetResourceRoot() + "/shader/color.fs").c_str());
			entity->AddComponent(shader);
		}

		{
			auto t = Neon::Time("Mesh Loading");

			auto entity = scene->CreateEntity("Entity/Mesh");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Mesh");
			entity->AddComponent(mesh);
			mesh->FromSTLFile(app.GetResourceRoot() + "/stl/mx.stl", 0.01f, 0.01f, 0.01f);
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

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", (app.GetResourceRoot() + "/shader/lighting.vs").c_str(), (app.GetResourceRoot() + "/shader/lighting.fs").c_str());
			entity->AddComponent(shader);

			auto transform = scene->CreateComponent<Neon::Transform>("Transform/Mesh");
			entity->AddComponent(transform);
			transform->AddUpdateCallback([transform](float now, float timeDelta) {
				//transform->rotation = glm::angleAxis(glm::radians(now * 0.01f), glm::vec3(0.0f, 1.0f, 0.0f));
				});
		}

		//{
		//	frame.AddVertex(-0.5f, -0.5f, 0.0f);
		//	frame.AddVertex(0.5f, -0.5f, 0.0f);
		//	frame.AddVertex(0.5f, 0.5f, 0.0f);
		//	frame.AddVertex(-0.5f, 0.5f, 0.0f);

		//	frame.AddIndex(0);
		//	frame.AddIndex(1);
		//	frame.AddIndex(2);

		//	frame.AddIndex(0);
		//	frame.AddIndex(2);
		//	frame.AddIndex(3);

		//	frame.AddUV(0.0f, 0.0f);
		//	frame.AddUV(1.0f, 0.0f);
		//	frame.AddUV(1.0f, 1.0f);
		//	frame.AddUV(0.0f, 1.0f);

		//	textureFrame = new Neon::Texture("frame", 1920, 1080);
		//	fbo = new Neon::FrameBufferObject("frame", textureFrame);
		//}

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
