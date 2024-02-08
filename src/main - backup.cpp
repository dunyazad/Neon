#include <iostream>

#include <Neon/Neon.h>

#include <Neon/CUDA/CUDATest.h>
#include <Neon/CUDA/CUDATSDF.h>
#include <Neon/CUDA/CUDASurfaceExtraction.h>

int gindex = 0;

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

#pragma region Toggler
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
							if (GLFW_MOD_SHIFT & event.mods)
							{
								mesh->ToggleFillMode();
								cout << "Toggle Fill Mode : " << mesh->GetName() << endl;
							}
							else
							{
								entity->ToggleVisible();
							}
						}
					}
				}
				});
		}
#pragma endregion

#pragma region Scene Event Handler
		{
			auto sceneEventHandler = scene->CreateEntity("Scene/EventHandler");
			sceneEventHandler->AddKeyEventHandler([scene](const Neon::KeyEvent& event) {
				if ((GLFW_KEY_KP_ADD == event.key || GLFW_KEY_EQUAL == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
					GLfloat pointSize;
					glGetFloatv(GL_POINT_SIZE, &pointSize);
					pointSize += 1.0f;
					glPointSize(pointSize);
				}
				else if ((GLFW_KEY_KP_SUBTRACT == event.key || GLFW_KEY_MINUS == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
					GLfloat pointSize;
					glGetFloatv(GL_POINT_SIZE, &pointSize);
					pointSize -= 1.0f;
					if (1.0f >= pointSize)
						pointSize = 1.0f;
					glPointSize(pointSize);
				}
				else if ((GLFW_KEY_KP_MULTIPLY == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
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
#pragma endregion

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

		size_t hResolution = 256;
		size_t vResolution = 480;
		float xUnit = 0.1f;
		float yUnit = 0.1f;
		float voxelSize = 0.1f;

		vector<Eigen::Matrix4f> transforms;
		vector<Neon::Mesh*> meshes;
		Neon::AABB meshesAABB;

		int currentModelIndex = 10;

		{
#pragma region Transform Info File
			auto tfile = ifstream();
			tfile.open("C:\\Resources\\MC_TESTDATA\\transform.txt", ios::in);
			std::string line;
			while (std::getline(tfile, line)) {
				//std::cout << "Read line: " << line << std::endl;
				stringstream ss(line);

				string word;
				ss >> word;
				Eigen::Matrix4f m;
				for (size_t r = 0; r < 4; r++)
				{
					for (size_t c = 0; c < 4; c++)
					{
						ss >> (*(m.data() + 4 * r + c));
					}
				}

				transforms.push_back(m);
			}
#pragma endregion
		}

#pragma region Model Local Axes
		{
			auto entity = scene->CreateEntity("Entity/Local Axes");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Local Axes");
			entity->AddComponent(mesh);

			mesh->SetDrawingMode(GL_LINES);
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(10.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 10.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 10.0f));

			mesh->AddColor(glm::vec4(1.0f, 0.5f, 0.5f, 1.0f));
			mesh->AddColor(glm::vec4(1.0f, 0.5f, 0.5f, 1.0f));
			mesh->AddColor(glm::vec4(0.5f, 1.0f, 0.5f, 1.0f));
			mesh->AddColor(glm::vec4(0.5f, 1.0f, 0.5f, 1.0f));
			mesh->AddColor(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);
			mesh->AddIndex(3);
			mesh->AddIndex(4);
			mesh->AddIndex(5);

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Color", Neon::URL::Resource("/shader/color.vs"), Neon::URL::Resource("/shader/color.fs"));
			entity->AddComponent(shader);

			Eigen::Matrix4f transformMatrix = transforms[currentModelIndex];
			auto transform = scene->CreateComponent<Neon::Transform>("Trnasform/Local Axes");
			entity->AddComponent(transform);
			transform->SetLocalTransform(glm::make_mat4(transformMatrix.data()));

			for (size_t i = 0; i < 16; i++)
			{
				cout << transformMatrix.data()[i] << endl;
			}
		}
#pragma endregion

#pragma region Draw Grid
		{
			size_t hResolution = 256;
			size_t vResolution = 480;

			float minX = -((float)hResolution * xUnit * 0.5f);
			float maxX = ((float)hResolution * xUnit * 0.5f);
			float minY = -((float)vResolution * yUnit * 0.5f);
			float maxY = ((float)vResolution * yUnit * 0.5f);

			for (float y = minY; y <= maxY; y += yUnit)
			{
				scene->Debug("grid lines")->AddLine({ minX, y, 0.0f }, { maxX, y, 0.0f }, glm::lightgray, glm::lightgray);
			}
			for (float x = minX; x <= maxX; x += xUnit)
			{
				scene->Debug("grid lines")->AddLine({ x, minY, 0.0f }, { x, maxY, 0.0f }, glm::lightgray, glm::lightgray);
			}
		}
		{
			float minX = -((float)hResolution * xUnit * 0.5f) + xUnit * 0.5f;
			float maxX = ((float)hResolution * xUnit * 0.5f) - xUnit * 0.5f;
			float minY = -((float)vResolution * yUnit * 0.5f) + yUnit * 0.5f;
			float maxY = ((float)vResolution * yUnit * 0.5f) - yUnit * 0.5f;

			for (float y = minY; y <= maxY; y += yUnit)
			{
				scene->Debug("grid lines")->AddLine({ minX, y, 0.0f }, { maxX, y, 0.0f }, glm::gray, glm::gray);
			}
			for (float x = minX; x <= maxX; x += xUnit)
			{
				scene->Debug("grid lines")->AddLine({ x, minY, 0.0f }, { x, maxY, 0.0f }, glm::gray, glm::gray);
			}
		}
#pragma endregion

		{
#pragma region Load Mesh
			//for (size_t i = 0; i < transforms.size(); i++)
			{
				char buffer[128];
				memset(buffer, 0, 128);

				sprintf_s(buffer, "C:\\Resources\\MC_TESTDATA\\%00004d_source.ply", currentModelIndex);

				auto mesh = scene->CreateComponent<Neon::Mesh>(buffer);

				mesh->FromPLYFile(buffer);

				for (size_t y = 0; y < 480 - 3; y += 3)
				{
					for (size_t x = 0; x < 256 - 2; x += 2)
					{
						auto i0 = 256 * y + x;
						auto i1 = 256 * y + x + 2;
						auto i2 = 256 * (y + 3) + x;
						auto i3 = 256 * (y + 3) + x + 2;

						mesh->AddIndex(i0);
						mesh->AddIndex(i1);
						mesh->AddIndex(i2);

						mesh->AddIndex(i2);
						mesh->AddIndex(i1);
						mesh->AddIndex(i3);

						//auto& v0 = mesh->GetVertex(i0);
						//auto& v1 = mesh->GetVertex(i1);
						//auto& v2 = mesh->GetVertex(i2);
						//auto& v3 = mesh->GetVertex(i3);

						//if ((FLT_VALID(v0.x) && FLT_VALID(v0.y) && FLT_VALID(v0.z)) &&
						//	(FLT_VALID(v1.x) && FLT_VALID(v1.y) && FLT_VALID(v1.z)) &&
						//	(FLT_VALID(v2.x) && FLT_VALID(v2.y) && FLT_VALID(v2.z)))
						//{
						//	scene->Debug("Triangles")->AddTriangle(v0, v2, v1);
						//}

						//if ((FLT_VALID(v2.x) && FLT_VALID(v2.y) && FLT_VALID(v2.z)) &&
						//	(FLT_VALID(v1.x) && FLT_VALID(v1.y) && FLT_VALID(v1.z)) &&
						//	(FLT_VALID(v3.x) && FLT_VALID(v3.y) && FLT_VALID(v3.z)))
						//{
						//	scene->Debug("Triangles")->AddTriangle(v2, v3, v1);
						//}
					}
				}
				meshes.push_back(mesh);

				//for (size_t y = 0; y < 480; y++)
				//{
				//	for (size_t x = 0; x < 256; x++)
				//	{
				//		auto& v = mesh->GetVertexBuffer()->GetElement(y * 256 + x);
				//		if (FLT_VALID(v.x) && FLT_VALID(v.y) && FLT_VALID(v.z))
				//		{
				//			printf("V [%d, %d] %f, %f, %f\n", x, y, v.x, v.y, v.z);
				//		}
				//	}
				//}

				//mesh->Bake(transforms[i]);

				// Debug Mesh
				scene->Debug(buffer)->AddMesh(mesh);

				auto a = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				auto b = Eigen::Vector3f(16.0f, 20.0f, 20.0f);
				printf("((b - a) %f\n", (b - a).norm());


				//auto transform = scene->CreateComponent<Neon::Transform>("deubg mesh transform");
				//transform->SetLocalTransform(glm::make_mat4(transforms[currentModelIndex].data()));
				//scene->Debug(buffer)->AddComponent(transform);
			}

			for (size_t i = 0; i < meshes.size(); i++)
			{
				auto& mesh = meshes[i];

				auto vmin3 = Eigen::Vector3f(mesh->GetAABB().GetMinPoint().x, mesh->GetAABB().GetMinPoint().y, mesh->GetAABB().GetMinPoint().z);
				auto vmax3 = Eigen::Vector3f(mesh->GetAABB().GetMaxPoint().x, mesh->GetAABB().GetMaxPoint().y, mesh->GetAABB().GetMaxPoint().z);

				Eigen::Vector4f vmin4 = transforms[i] * Eigen::Vector4f(vmin3.x(), vmin3.y(), vmin3.z(), 1.0f);
				Eigen::Vector4f vmax4 = transforms[i] * Eigen::Vector4f(vmax3.x(), vmax3.y(), vmax3.z(), 1.0f);
				meshesAABB.Expand(glm::vec3(vmin4.x(), vmin4.y(), vmin4.z()));
				meshesAABB.Expand(glm::vec3(vmax4.x(), vmax4.y(), vmax4.z()));
			}
#pragma endregion

			//NeonCUDA::BuildDepthMapTest(scene, meshes[0]);

			//return;
			{
				NeonCUDA::SurfaceExtractor se;
				se.NewFrameWrapper(scene, meshes[0], transforms[currentModelIndex]);

				return;
			}
		}

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
