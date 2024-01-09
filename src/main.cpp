#include <iostream>

#include <Neon/Neon.h>

#include <Neon/CUDA/CUDATest.h>
#include <Neon/CUDA/CUDATSDF.h>
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

			mesh->AddColor(glm::vec4(0.0f, 1.0f, 1.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 1.0f, 1.0f, 1.0f));
			mesh->AddColor(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f));
			mesh->AddColor(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f));
			mesh->AddColor(glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);
			mesh->AddIndex(3);
			mesh->AddIndex(4);
			mesh->AddIndex(5);

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Color", Neon::URL::Resource("/shader/color.vs"), Neon::URL::Resource("/shader/color.fs"));
			entity->AddComponent(shader);

			glm::mat4 transformMatrix(
				9.999999E-01, -3.041836E-04, 5.392341E-05, 0.000000E+00,
				3.043024E-04, 9.999977E-01, -2.120121E-03, 0.000000E+00,
				-5.327794E-05, 2.120084E-03, 9.999976E-01, 0.000000E+00,
				9.976241E-02, 1.964474E-01, -4.013956E-01, 1.000000E+00);

			auto transform = scene->CreateComponent<Neon::Transform>("Trnasform/Local Axes");
			entity->AddComponent(transform);
			transform->SetLocalTransform(transformMatrix);
		}
#pragma endregion

		{
			auto entity = scene->CreateEntity("Entity/RegularGrid");
			entity->AddKeyEventHandler([scene, entity](const Neon::KeyEvent& event) {
				if (GLFW_KEY_ESCAPE == event.key && GLFW_RELEASE == event.action) {
					auto mesh = entity->GetComponent<Neon::Mesh>(0);
						if (nullptr != mesh)
						{
							mesh->ToggleFillMode();
								cout << "Toggle Fill Mode : " << mesh->GetName() << endl;
						}
				}
			});
		
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/PLY Input");
			entity->AddComponent(mesh);
			//mesh->FromPLYFile("C:\\saveData\\0000_target.ply");
			mesh->FromPLYFile("C:\\Resources\\MC_TESTDATA\\0000_1st_TargetPC.ply");

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			glm::mat4 transformMatrix(
				9.999999E-01, -3.041836E-04, 5.392341E-05, 0.000000E+00,
				3.043024E-04, 9.999977E-01, -2.120121E-03, 0.000000E+00,
				-5.327794E-05, 2.120084E-03, 9.999976E-01, 0.000000E+00,
				9.976241E-02, 1.964474E-01, - 4.013956E-01, 1.000000E+00);

			for (size_t y = 0; y < 480 - 3; y+=3)
			{
				for (size_t x = 0; x < 256 - 2; x+=2)
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

			scene->Debug("Mesh")->AddMesh(mesh);

			mesh->Bake(transformMatrix);
			mesh->FillColor(glm::green);

			auto& minPoint = mesh->GetAABB().GetMinPoint();
			auto& maxPoint = mesh->GetAABB().GetMaxPoint();

			float xoffset = (maxPoint.x - minPoint.x) * 0.5f;
			float yoffset = (maxPoint.y - minPoint.y) * 0.5f;
			float zoffset = (maxPoint.z - minPoint.z) * 0.5f;

			int xcount = 2;
			int ycount = 2;
			int zcount = 2;

			float voxelSize = 0.05f;
			float isoValue = 2.5f;

			NeonCUDA::TSDF** tsdfs = new NeonCUDA::TSDF * [xcount * ycount * zcount];
			for (size_t z = 0; z < zcount; z++)
			{
				for (size_t y = 0; y < ycount; y++)
				{
					for (size_t x = 0; x < xcount; x++)
					{
						tsdfs[z * ycount * xcount + y * xcount + x] = new NeonCUDA::TSDF(
							voxelSize,
							make_float3(minPoint.x + x * xoffset, minPoint.y + y * yoffset, minPoint.z + z * zoffset),
							make_float3(minPoint.x + (x + 1) * xoffset, minPoint.y + (y + 1) * yoffset, minPoint.z + (z + 1) * zoffset));
					}
				}
			}

			auto theO = glm::vec3(transformMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
			mesh->GetAABB().Clear();
			for (auto& v : mesh->GetVertexBuffer()->GetElements())
			{
				v.z = theO.z;

				mesh->GetAABB().Expand(v);
			}
			mesh->GetVertexBuffer()->SetDirty();

			scene->Debug("AABB")->AddAABB(mesh->GetAABB());


			cudaDeviceSynchronize();

			nvtxRangePushA("@Aaron/Total");

			for (size_t i = 0; i < 1; i++)
			{
				nvtxRangePushA("@Aaron/UpdateValues - Total");
				for (size_t z = 0; z < zcount; z++)
				{
					for (size_t y = 0; y < ycount; y++)
					{
						for (size_t x = 0; x < xcount; x++)
						{
							tsdfs[z * ycount * xcount + y * xcount + x]->UpdateValues();
						}
					}
				}
				nvtxRangePop();

				nvtxRangePushA("@Aaron/BuildGridCells - Total");
				for (size_t z = 0; z < zcount; z++)
				{
					for (size_t y = 0; y < ycount; y++)
					{
						for (size_t x = 0; x < xcount; x++)
						{
							tsdfs[z * ycount * xcount + y * xcount + x]->BuildGridCells(isoValue);
						}
					}
				}
				nvtxRangePop();
			}

			nvtxRangePushA("@Aaron/TestTriangles - Total");

			for (size_t z = 0; z < zcount; z++)
			{
				for (size_t y = 0; y < ycount; y++)
				{
					for (size_t x = 0; x < xcount; x++)
					{
						tsdfs[z * ycount * xcount + y * xcount + x]->TestTriangles(scene);
					}
				}
			}
			nvtxRangePop();

			nvtxRangePop();
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
