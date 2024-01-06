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

		{
			auto entity = scene->CreateEntity("Entity/RegularGrid");

			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/PLY Input");
			entity->AddComponent(mesh);
			mesh->FromPLYFile("C:\\saveData\\0000_target.ply");

			auto& minPoint = mesh->GetAABB().GetMinPoint();
			auto& maxPoint = mesh->GetAABB().GetMaxPoint();

			float xoffset = (maxPoint.x - minPoint.x) * 0.5f;
			float yoffset = (maxPoint.y - minPoint.y) * 0.5f;
			float zoffset = (maxPoint.z - minPoint.z) * 0.5f;

			int xcount = 4;
			int ycount = 4;
			int zcount = 4;

			float voxelSize = 0.05f;
		
			NeonCUDA::TSDF** tsdfs = new NeonCUDA::TSDF*[xcount * ycount * zcount];
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

			for (size_t z = 0; z < zcount; z++)
			{
				for (size_t y = 0; y < ycount; y++)
				{
					for (size_t x = 0; x < xcount; x++)
					{
						tsdfs[z * ycount * xcount + y * xcount + x]->Test(scene);
					}
				}
			}

			//NeonCUDA::TSDF tsdf(
			//	0.05f,
			//	make_float3(minPoint.x, minPoint.y, minPoint.z),
			//	make_float3(maxPoint.x, maxPoint.y, maxPoint.z));

			////tsdf.Apply(mesh);

			//tsdf.UpdateValues();

			//tsdf.Test(scene);

			//auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			//entity->AddComponent(shader);

			//auto grid = scene->CreateComponent<Neon::RegularGrid>("Volume", mesh, 0.5f);
			//grid->Build();

			//for (size_t z = 0; z < grid->GetCellCountZ(); z++)
			//{
			//	for (size_t y = 0; y < grid->GetCellCountY(); y++)
			//	{
			//		for (size_t x = 0; x < grid->GetCellCountX(); x++)
			//		{
			//			auto cell = grid->GetCell(make_tuple(x, y, z));
			//			if (cell->GetTriangles().size() > 0)
			//			{
			//				printf("[%d, %d, %d] Triangle Exists\n", x, y, z);
			//			}
			//		}
			//	}
			//}

			//auto triangleLists = grid->ExtractSurface(0.0f);

			//cout << mesh->GetAABB() << endl;

#pragma region Temp
			//auto not = mesh->GetIndexBuffer()->Size() / 3;
			//
			//			float gridInterval = 0.5f;
			//
			//			auto xLength = mesh->GetAABB().GetXLength();
			//			auto yLength = mesh->GetAABB().GetYLength();
			//			auto zLength = mesh->GetAABB().GetZLength();
			//
			//			auto xCount = (int)ceilf(xLength / gridInterval) + 1;
			//			auto yCount = (int)ceilf(yLength / gridInterval) + 1;
			//			auto zCount = (int)ceilf(zLength / gridInterval) + 1;
			//
			//			float* grid = new float[xCount * yCount * zCount];
			//			memset(grid, 0, sizeof(float) * xCount * yCount * zCount);
			//
			//			float nx = -((float)xCount * gridInterval) * 0.5f;
			//			float px = ((float)xCount * gridInterval) * 0.5f;
			//			float ny = -((float)yCount * gridInterval) * 0.5f;
			//			float py = ((float)yCount * gridInterval) * 0.5f;
			//			float nz = -((float)zCount * gridInterval) * 0.5f;
			//			float pz = ((float)zCount * gridInterval) * 0.5f;
			//			float cx = (px + nx) * 0.5f;
			//			float cy = (py + ny) * 0.5f;
			//			float cz = (pz + nz) * 0.5f;
			//
			//			Neon::AABB aabb;
			//			aabb.Expand(glm::vec3(nx, ny, nz) + mesh->GetAABB().GetCenter());
			//			aabb.Expand(glm::vec3(px, py, pz) + mesh->GetAABB().GetCenter());
			//
			//#pragma omp parallel for
			//			for (int z = 0; z < zCount; z++)
			//			{
			//#pragma omp parallel for
			//				for (int y = 0; y < yCount; y++)
			//				{
			//#pragma omp parallel for
			//					for (int x = 0; x < xCount; x++)
			//					{
			//						float minDistance = FLT_MAX;
			//
			//#pragma omp parallel for
			//						for (int i = 0; i < not; i++)
			//						{
			//							auto i0 = mesh->GetIndex(i * 3 + 0);
			//							auto i1 = mesh->GetIndex(i * 3 + 1);
			//							auto i2 = mesh->GetIndex(i * 3 + 2);
			//
			//							auto& v0 = mesh->GetVertex(i0);
			//							auto& v1 = mesh->GetVertex(i1);
			//							auto& v2 = mesh->GetVertex(i2);
			//
			//							auto point = glm::vec3(nx, ny, nz) + mesh->GetAABB().GetCenter() + glm::vec3(x * gridInterval, y * gridInterval, z * gridInterval);
			//							auto normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
			//							auto vectorToVertex = point - v0;
			//
			//							float distance = glm::dot(normal, vectorToVertex) / glm::length(normal);
			//
			//							if (minDistance > distance)
			//							{
			//								minDistance = distance;
			//							}
			//						}
			//
			//						//if (fabsf(minDistance) < 2.0f)
			//						{
			//							grid[z * yCount * xCount + y * xCount + x] = minDistance;
			//
			//							//printf("%d %d %d %f\n", x, y, z, minDistance);
			//						}
			//					}
			//				}
			//			}
			//
			//			for (int z = 0; z < zCount; z++)
			//			{
			//				for (int y = 0; y < yCount; y++)
			//				{
			//					for (int x = 0; x < xCount; x++)
			//					{
			//						auto value = grid[z * yCount * xCount + y * xCount + x];
			//
			//						if (value != 0)
			//						{
			//							printf("%d %d %d %f\n", x, y, z, value);
			//						}
			//					}
			//				}
			//			}  
#pragma endregion


			scene->Debug("Mesh")->AddMesh(mesh);
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
