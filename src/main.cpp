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

		//glEnable(GL_CULL_FACE);
		//glCullFace(GL_BACK);
		//glFrontFace(GL_CCW);

		glPointSize(10.0f);

		auto scene = app.CreateScene("Scene/Main");

#pragma region For Debugging
		auto toggleFillMode1 = [](const Neon::KeyEvent& event)
			{
				auto entity = dynamic_cast<Neon::Entity*>(event.target);
				if (nullptr != entity)
				{
					auto mesh = entity->GetComponent<Neon::Mesh>(0);
					if (nullptr != mesh)
					{
						if (GLFW_KEY_1 == event.key && GLFW_RELEASE == event.action)
						{
							mesh->ToggleFillMode();
						}
					}
				}
			};
		auto toggleFillMode2 = [](const Neon::KeyEvent& event)
			{
				auto entity = dynamic_cast<Neon::Entity*>(event.target);
				if (nullptr != entity)
				{
					auto mesh = entity->GetComponent<Neon::Mesh>(0);
					if (nullptr != mesh)
					{
						if (GLFW_KEY_2 == event.key && GLFW_RELEASE == event.action)
						{
							mesh->ToggleFillMode();
						}
					}
				}
			};
		auto toggleFillMode3 = [](const Neon::KeyEvent& event)
			{
				auto entity = dynamic_cast<Neon::Entity*>(event.target);
				if (nullptr != entity)
				{
					auto mesh = entity->GetComponent<Neon::Mesh>(0);
					if (nullptr != mesh)
					{
						if (GLFW_KEY_3 == event.key && GLFW_RELEASE == event.action)
						{
							mesh->ToggleFillMode();
						}
					}
				}
			};
		auto toggleFillMode4 = [](const Neon::KeyEvent& event)
			{
				auto entity = dynamic_cast<Neon::Entity*>(event.target);
				if (nullptr != entity)
				{
					auto mesh = entity->GetComponent<Neon::Mesh>(0);
					if (nullptr != mesh)
					{
						if (GLFW_KEY_4 == event.key && GLFW_RELEASE == event.action)
						{
							mesh->ToggleFillMode();
						}
					}
				}
			};

		auto toggleFillMode5 = [](const Neon::KeyEvent& event)
			{
				auto entity = dynamic_cast<Neon::Entity*>(event.target);
				if (nullptr != entity)
				{
					auto mesh = entity->GetComponent<Neon::Mesh>(0);
					if (nullptr != mesh)
					{
						if (GLFW_KEY_5 == event.key && GLFW_RELEASE == event.action)
						{
							mesh->ToggleFillMode();
						}
					}
				}
			};

		auto debugPoints = scene->CreateDebugEntity("DebugEntity/Points");
		debugPoints->AddKeyEventHandler(toggleFillMode2);

		auto debugLines = scene->CreateDebugEntity("DebugEntity/Lines");
		debugLines->AddKeyEventHandler(toggleFillMode3);

		auto debugTriangles = scene->CreateDebugEntity("DebugEntity/Triangles");
		debugTriangles->AddKeyEventHandler(toggleFillMode4);

		auto debugBoxes = scene->CreateDebugEntity("DebugEntity/Boxes");
		debugBoxes->AddKeyEventHandler(toggleFillMode5);
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
		
		/*
		{
			auto entity = scene->CreateEntity("Entity/spot");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/spot");
			entity->AddComponent(mesh);

			mesh->FromPLYFile(Neon::URL("C:/Resources/TestData/OCT_phantom_1.ply"));
			cout << "loaded" << endl;

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			entity->AddKeyEventHandler([mesh](const Neon::KeyEvent& event) {
				if (GLFW_KEY_1 == event.key && GLFW_RELEASE == event.action)
				{
					mesh->ToggleFillMode();
				}
				});

			auto nov = mesh->GetVertexBuffer()->Size();
			for (size_t i = 0; i < nov; i++)
			{
				auto v = mesh->GetVertex(i);
				auto c = mesh->GetColor(i);
				debugPoints->AddPoint(v, c);
			}
		}
		*/

		{
			auto entity = scene->CreateEntity("Entity/spot");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/spot");
			entity->AddComponent(mesh);

			mesh->FromSTLFile(Neon::URL::Resource("/stl/sphere.stl"));
			mesh->FillColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
			mesh->RecalculateFaceNormal();

			scene->GetMainCamera()->centerPosition = mesh->GetAABB().GetCenter();
			scene->GetMainCamera()->distance = mesh->GetAABB().GetDiagonalLength();

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			auto transform = scene->CreateComponent<Neon::Transform>("Transform/spot");
			entity->AddComponent(transform);
			entity->AddKeyEventHandler([mesh](const Neon::KeyEvent& event) {
				if (GLFW_KEY_1 == event.key && GLFW_RELEASE == event.action)
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
				});

			entity->AddMouseButtonEventHandler([scene, mesh, debugPoints, debugLines](const Neon::MouseButtonEvent& event) {
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
						//debugPoints->Clear();

						//debugPoints->AddPoint(intersection, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
					}
				}
				});

			auto t0 = Neon::Time("Regular Grid");
			auto cellSize = 0.25f;
			auto regularGrid = scene->CreateComponent<Neon::RegularGrid>("RegularGrid/spot", mesh, cellSize);
			entity->AddComponent(regularGrid);
			regularGrid->Build();

			struct GridCell
			{
				glm::vec3 vertex[8];
				float value[8];
			};

			auto cells = regularGrid->GetCells();
			for (size_t z = 0; z < regularGrid->GetCellCountZ(); z++)
			{
				for (size_t y = 0; y < regularGrid->GetCellCountY(); y++)
				{
					for (size_t x = 0; x < regularGrid->GetCellCountX(); x++)
					{
						auto cell = cells[z][y][x];
						if (0 < cell->GetTriangles().size())
						{
							if (x == 79 && y == 22 && z == 20)
							{
								debugBoxes->AddBox(cell->GetCenter(), cell->GetXLength(), cell->GetYLength(), cell->GetZLength(), glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
							}

							glm::vec3 planePoint = glm::zero<glm::vec3>();
							glm::vec3 planeNormal = glm::zero<glm::vec3>();
							for (auto& t : cell->GetTriangles())
							{
								auto v0 = mesh->GetVertex(t->v0->index);
								auto v1 = mesh->GetVertex(t->v1->index);
								auto v2 = mesh->GetVertex(t->v2->index);

								auto n = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
								auto c = (v0 + v1 + v2) / 3.0f;

								planePoint += c;
								planeNormal += n;
							}
							
							planePoint /= (float)cell->GetTriangles().size();
							planeNormal /= (float)cell->GetTriangles().size();
							planeNormal = normalize(planeNormal);

							GridCell gridCell;
							gridCell.vertex[0] = cell->xyz;
							gridCell.vertex[1] = cell->Xyz;
							gridCell.vertex[2] = cell->XyZ;
							gridCell.vertex[3] = cell->xyZ;
							gridCell.vertex[4] = cell->xYz;
							gridCell.vertex[5] = cell->XYz;
							gridCell.vertex[6] = cell->XYZ;
							gridCell.vertex[7] = cell->xYZ;
							gridCell.value[0] = 1.0f;
							gridCell.value[1] = 1.0f;
							gridCell.value[2] = 1.0f;
							gridCell.value[3] = 1.0f;
							gridCell.value[4] = 1.0f;
							gridCell.value[5] = 1.0f;
							gridCell.value[6] = 1.0f;
							gridCell.value[7] = 1.0f;

							for (size_t i = 0; i < 8; i++)
							{
								if (0 < glm::dot(gridCell.vertex[i] - planePoint, planeNormal))
								{
									debugPoints->AddPoint(gridCell.vertex[i], glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
								}
								else if (0 > glm::dot(gridCell.vertex[i] - planePoint, planeNormal))
								{
									//debugPoints->AddPoint(planePoint, glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
								}
								else
								{
									//debugPoints->AddPoint(planePoint, glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
								}
							}

							if (x == 79 && y == 22 && z == 20)
							{
								//debugPoints->AddPoint(planePoint, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
								debugLines->AddLine(planePoint, planePoint + planeNormal * 0.125f, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));


							}
						}
					}
				}
			}

			//regularGrid->AddMouseButtonEventHandler([scene, mesh, regularGrid, debugPoints, debugLines](const Neon::MouseButtonEvent& event) {
			//	if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_RELEASE)
			//	{
			//		auto camera = scene->GetMainCamera();

			//		auto ray = camera->GetPickingRay(event.xpos, event.ypos);

			//		glm::vec3 intersection;
			//		size_t faceIndex = 0;
			//		if (mesh->Pick(ray, intersection, faceIndex))
			//		{
			//			auto index = regularGrid->GetIndex(intersection);

			//			cout << "x : " << get<0>(index) << " , y : " << get<1>(index) << " , z : " << get<2>(index) << endl;
			//		}
			//	}
			//	});

			auto result = regularGrid->ExtractSurface(0.0f);
			for (auto& vs : result)
			{
				debugTriangles->AddTriangle(vs[0], vs[1], vs[2], glm::vec4(0.7f, 0.6f, 0.4f, 1.0f), glm::vec4(0.7f, 0.6f, 0.4f, 1.0f), glm::vec4(0.7f, 0.6f, 0.4f, 1.0f));
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

		SAFE_DELETE(imageB);
		SAFE_DELETE(imageC);

		});

	app.Run();

	return 0;
}
