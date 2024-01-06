#include <iostream>

#include <Neon/Neon.h>

#include <Neon/CUDA/CUDATest.h>
#include "DT/DT.h"

int main()
{
	Neon::Application app(1280, 1024);
	Neon::URL::ChangeDirectory("..");

	//app.GetWindow()->UseVSync(false);

	int index = 0;

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		//glEnable(GL_CULL_FACE);
		//glCullFace(GL_BACK);
		//glFrontFace(GL_CCW);

		glPointSize(10.0f);

		auto scene = app.CreateScene("Scene/Main");
		{
			auto toggler = scene->CreateEntity("Scene/Toggler");
			toggler->AddKeyEventHandler([scene](const Neon::KeyEvent& event) {
				if ((GLFW_KEY_0 <= event.key && GLFW_KEY_9 >= event.key) && GLFW_RELEASE == event.action) {
					auto debugEntities = scene->GetDebugEntities();
					if (event.key - GLFW_KEY_0 < debugEntities.size())
					{
						auto entity = debugEntities[(event.key - GLFW_KEY_0 + 1) % debugEntities.size()];

						auto mesh = entity->GetComponent<Neon::Mesh>(0);
						if (nullptr != mesh)
						{
							mesh->ToggleFillMode();
							cout << "Toggle Fill Mode : " << mesh->GetName() << endl;
						}
					}
				}
				});
		}
		{
			auto sceneEventHandler = scene->CreateEntity("Scene/EventHandler");
			sceneEventHandler->AddKeyEventHandler([scene](const Neon::KeyEvent& event) {
				if ((GLFW_KEY_KP_ADD == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
					GLfloat pointSize;
					glGetFloatv(GL_POINT_SIZE, &pointSize);
					pointSize += 1.0f;
					glPointSize(pointSize);
				}
				else if ((GLFW_KEY_KP_SUBTRACT == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
					GLfloat pointSize;
					glGetFloatv(GL_POINT_SIZE, &pointSize);
					pointSize -= 1.0f;
					if (1.0f >= pointSize)
						pointSize = 1.0f;
					glPointSize(pointSize);
				} else if ((GLFW_KEY_KP_MULTIPLY == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
					GLfloat lineWidth;
					glGetFloatv(GL_LINE_WIDTH, &lineWidth);
					lineWidth += 1.0f;
					glLineWidth(lineWidth);
				}
				else if ((GLFW_KEY_KP_DIVIDE == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
					GLfloat lineWidth;
					glGetFloatv(GL_LINE_WIDTH, &lineWidth);
					lineWidth -= 1.0f;
					if (1.0f >= lineWidth)
						lineWidth = 1.0f;
					glLineWidth(lineWidth);
				}
				});
		}

#pragma region Camera
		{
			auto entity = scene->CreateEntity("Entity/Main Camera");
			auto transform = scene->CreateComponent<Neon::Transform>("Transform/Main Camera");
			entity->AddComponent(transform);

			auto camera = scene->CreateComponent<Neon::Camera>("Camera/Main", 1280.0f, 1024.0f);
			entity->AddComponent(camera);
			camera->distance = 2.0f;
			camera->centerPosition = glm::vec3(0.0f, 0.0f, 0.0f);
			camera->zFar = 10000.0f;
			scene->SetMainCamera(camera);

			auto cameraManipulator = scene->CreateComponent<Neon::CameraManipulator>("CameraManipulator/Main", entity, camera);
			entity->AddComponent(cameraManipulator);
		}
#pragma endregion

#pragma region Light
		{
			auto entity = scene->CreateEntity("Entity/Main Light");

			auto light = scene->CreateComponent<Neon::Light>("Light/Main");
			entity->AddComponent(light);

			light->AddUpdateHandler([scene, light](double now, double timeDelta) {
				auto camera = scene->GetMainCamera();
				light->position = camera->position;
				light->direction = glm::normalize(camera->centerPosition - camera->position);
				});

			scene->SetMainLight(light);
		}
#pragma endregion

#pragma region Guide Axes
		{
			auto entity = scene->CreateEntity("Entity/Axes");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Axes");
			entity->AddComponent(mesh);

			mesh->SetDrawingMode(GL_LINES);
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(10.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 10.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 10.0f));

			mesh->AddColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);
			mesh->AddIndex(3);
			mesh->AddIndex(4);
			mesh->AddIndex(5);

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Color", Neon::URL::Resource("/shader/color.vs"), Neon::URL::Resource("/shader/color.fs"));
			entity->AddComponent(shader);
		}
#pragma endregion

#pragma region Load Model
		{
			auto entity = scene->CreateEntity("Entity/Model");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Model");
			entity->AddComponent(mesh);

			mesh->FromSTLFile("C:\\Resources\\3D\\STL\\a.stl");
			mesh->FillColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
			mesh->RecalculateFaceNormal();

			scene->GetMainCamera()->centerPosition = mesh->GetAABB().GetCenter();
			scene->GetMainCamera()->distance = mesh->GetAABB().GetDiagonalLength();

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			entity->AddKeyEventHandler([mesh](const Neon::KeyEvent& event) {
				if (GLFW_KEY_ESCAPE == event.key && GLFW_RELEASE == event.action)
				{
					mesh->ToggleFillMode();
				}
				else if (GLFW_KEY_KP_ADD == event.key && (GLFW_PRESS == event.action || GLFW_REPEAT == event.action))
				{
					GLfloat currentPointSize;
					glGetFloatv(GL_POINT_SIZE, &currentPointSize);
					glPointSize(currentPointSize + 1.0f);
				}
				else if (GLFW_KEY_KP_SUBTRACT == event.key && (GLFW_PRESS == event.action || GLFW_REPEAT == event.action))
				{
					GLfloat currentPointSize;
					glGetFloatv(GL_POINT_SIZE, &currentPointSize);
					glPointSize(currentPointSize - 1.0f);
				}
				else if (GLFW_KEY_S == event.key && GLFW_RELEASE == event.action)
				{
					mesh->ToSTLFile("C:\\Users\\USER\\Desktop\\result.stl");
				}
				});

			entity->AddMouseButtonEventHandler([entity, scene, mesh](const Neon::MouseButtonEvent& event) {
				if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_DOUBLE_ACTION)
				{
					auto camera = scene->GetMainCamera();

					auto ray = camera->GetPickingRay(event.xpos, event.ypos);

					glm::vec3 intersection;
					size_t faceIndex = 0;
					if (mesh->Pick(ray, intersection, faceIndex))
					{
						camera->centerPosition = intersection;
					}
				}
				else if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_RELEASE)
				{
					auto camera = scene->GetMainCamera();

					auto ray = camera->GetPickingRay(event.xpos, event.ypos);

					glm::vec3 intersection;
					size_t faceIndex = 0;
					if (mesh->Pick(ray, intersection, faceIndex))
					{
						scene->Debug("Points")->Clear();
						scene->Debug("Points")->AddPoint(intersection, glm::red);
					}
				}
				});
		}
#pragma endregion

#pragma region Grab
		{
			auto entity = scene->CreateEntity("Entity/Grab");
			auto transform = scene->CreateComponent<Neon::Transform>("Transform/Grab");
			entity->AddComponent(transform);

			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Grab");
			entity->AddComponent(mesh);
			
			int hres = 256;
			int vres = 480;
			float width = 4.0f;
			float height = 3.0f;

			float centerX = width * 0.5f;
			float centerY = height * 0.5f;

			float* depthValues = new float[hres * vres];

			for (size_t h = 0; h < vres; h++)
			{
				for (size_t w = 0; w < hres; w++)
				{
					float x = w * width / (float)hres - centerX;
					float y = h * height / (float)vres - centerY;
					mesh->AddVertex({ x, y, 0.0f });

					if (w < hres - 1 && h < vres - 1)
					{
						mesh->AddIndex(hres * h + w);
						mesh->AddIndex(hres * h + w + 1);
						mesh->AddIndex(hres * (h + 1) + w);

						mesh->AddIndex(hres * h + w + 1);
						mesh->AddIndex(hres * (h + 1) + w);
						mesh->AddIndex(hres * (h + 1) + w + 1);
					}
				}
			}

			mesh->FillColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
			mesh->RecalculateFaceNormal();

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			entity->AddKeyEventHandler([entity, transform, scene, mesh](const Neon::KeyEvent& event) {
				if (GLFW_KEY_ESCAPE == event.key && GLFW_RELEASE == event.action)
				{
					mesh->ToggleFillMode();
				}
				if (GLFW_KEY_ENTER == event.key && GLFW_RELEASE == event.action)
				{
					auto camera = scene->GetMainCamera();

					transform->rotation = glm::quat_cast(glm::inverse(camera->viewMatrix));
					transform->position = -glm::column(camera->viewMatrix, 3);

					transform->OnUpdate(0, 0);

					auto tm = transform->absoluteTransform;
					tm[2] = -tm[2];

					auto l0 = glm::vec3(tm * glm::vec4(0, 0, 0, 1.0f));
					auto l1 = glm::vec3(tm * glm::vec4(0, 0, 5, 1.0f));

					scene->Debug("Line")->AddLine(l0, l1, glm::red, glm::red);
				}
				});

			entity->AddMouseButtonEventHandler([entity, scene, mesh](const Neon::MouseButtonEvent& event) {
				if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_DOUBLE_ACTION)
				{
					auto camera = scene->GetMainCamera();

					auto ray = camera->GetPickingRay(event.xpos, event.ypos);

					glm::vec3 intersection;
					size_t faceIndex = 0;
					if (mesh->Pick(ray, intersection, faceIndex))
					{
						camera->centerPosition = intersection;
					}
				}
				else if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_RELEASE)
				{
					auto camera = scene->GetMainCamera();

					auto ray = camera->GetPickingRay(event.xpos, event.ypos);

					glm::vec3 intersection;
					size_t faceIndex = 0;
					if (mesh->Pick(ray, intersection, faceIndex))
					{
						scene->Debug("Points")->Clear();
						scene->Debug("Points")->AddPoint(intersection, glm::red);
					}
				}
				});
		}
#pragma endregion
	});



	app.OnUpdate([&](double now, double timeDelta) {
		//glPointSize(cosf(now * 0.005f) * 10.0f + 10.0f);

		//auto t = Neon::Time("Update");

		//fbo->Bind();

		//glClearColor(0.9f, 0.7f, 0.5f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//fbo->Unbind();

		glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		});






	app.OnTerminate([&]() {
		auto t = Neon::Time("Terminate");
		});

	app.Run();

	return 0;
}
