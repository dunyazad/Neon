#include <iostream>

#include <Neon/Neon.h>

#include "MC33.h"
#include "MC33.cpp"
#include "surface.cpp"
#include "grid3d.cpp"










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

			mesh->FromSTLFile(Neon::URL::Resource("/stl/mesh.stl"));
			mesh->FillColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
			mesh->RecalculateFaceNormal();

			/*
			{
				float offsetDistance = -0.1f;

				auto noi = mesh->GetIndexBuffer()->Size() / 3;

				vector<glm::vec3> vnormals(mesh->GetVertexBuffer()->Size());
				vector<int> vnormalRefs(mesh->GetVertexBuffer()->Size());

				for (size_t i = 0; i < noi; i++)
				{
					GLuint i0, i1, i2;
					mesh->GetTriangleVertexIndices(i, i0, i1, i2);

					auto v0 = mesh->GetVertex(i0);
					auto v1 = mesh->GetVertex(i1);
					auto v2 = mesh->GetVertex(i2);

					auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
					vnormals[i0] += normal;
					vnormals[i1] += normal;
					vnormals[i2] += normal;

					vnormalRefs[i0] += 1;
					vnormalRefs[i1] += 1;
					vnormalRefs[i2] += 1;

					scene->Debug("original")->AddTriangle(v0, v1, v2);
				}

				for (size_t i = 0; i < vnormals.size(); i++)
				{
					auto v = mesh->GetVertex(i);
					auto tv = v + vnormals[i] / (float)vnormalRefs[i] * offsetDistance;
					mesh->SetVertex(i, tv);
				}
			}
			*/

			scene->GetMainCamera()->centerPosition = mesh->GetAABB().GetCenter();
			scene->GetMainCamera()->distance = mesh->GetAABB().GetDiagonalLength();

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			auto transform = scene->CreateComponent<Neon::Transform>("Transform/spot");
			entity->AddComponent(transform);
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
						//debugPoints->Clear();

						//debugPoints->AddPoint(intersection, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

						auto vetm = entity->GetComponent<Neon::VETM>();
						auto vertex = vetm->GetNearestVertex(intersection);

						scene->Debug("Points")->Clear();
						scene->Debug("Points")->AddPoint(vertex->p, glm::red);

						for (auto& e : vertex->edges)
						{
							if (2 > e->triangles.size())
							{
								scene->Debug("Lines")->Clear();
								scene->Debug("Lines")->AddLine(e->v0->p, e->v1->p, glm::green, glm::green);
							}

							cout << e->id << " : " << e->triangles.size() << endl;
						}
					}
				}
				});


			{
				auto vetm = scene->CreateComponent<Neon::VETM>("VETM/Mesh", mesh);
				entity->AddComponent(vetm);
				vetm->Build();
				//{
				//	set<Neon::VETM::Edge*> borderEdges;

				//	for (auto& edge : vetm->GetEdges())
				//	{
				//		if (edge->triangles.size() < 2)
				//		{
				//			//borderEdges.insert(edge);
				//			scene->Debug("Border")->AddLine(edge->v0->p, edge->v1->p, glm::red, glm::red);
				//		}
				//	}
				//}

				//{
				//	Neon::Time("GetBorderEdges");

				//	auto foundBorderEdges = vetm->GetBorderEdges();

				//	cout << "Found Border Edges : " << foundBorderEdges.size() << endl;
				//}

				//{
				//	Neon::Time("GenerateBase");

				//	vetm->GenerateBase();
				//}

				{
					Neon::Time("Visulaize VETM");

					for (auto& t : vetm->GetTriangles())
					{
						scene->Debug("VETM")->AddTriangle(t->v0->p, t->v1->p, t->v2->p, glm::darkgray, glm::darkgray, glm::darkgray);
					}
				}
			}

			return;


			Neon::Time("Regular Grid");
			auto trimin = Trimin(mesh->GetAABB().GetXLength(), mesh->GetAABB().GetYLength(), mesh->GetAABB().GetZLength());
			auto cellSize = trimin * 0.005f;
			auto regularGrid = scene->CreateComponent<Neon::RegularGrid>("RegularGrid/spot", mesh, cellSize);
			entity->AddComponent(regularGrid);

			regularGrid->Build();

			//regularGrid->ForEachCell([scene](Neon::RGCell* cell, int x, int y, int z) {
			//	if (0 < cell->GetTriangles().size())
			//	{
			//		scene->Debug("Cells")->AddBox(cell->GetCenter(), cell->GetXLength(), cell->GetYLength(), cell->GetZLength(), glm::blue);
			//	}
			//	});

			regularGrid->AddMouseButtonEventHandler([scene, mesh, regularGrid](const Neon::MouseButtonEvent& event) {
				if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_RELEASE)
				{
					auto camera = scene->GetMainCamera();

					auto ray = camera->GetPickingRay(event.xpos, event.ypos);

					glm::vec3 intersection;
					size_t faceIndex = 0;
					if (mesh->Pick(ray, intersection, faceIndex))
					{
						auto index = regularGrid->GetIndex(intersection);

						cout << "x : " << get<0>(index) << " , y : " << get<1>(index) << " , z : " << get<2>(index) << endl;
					}
				}
				});

			/*
			{
				//grid3d G;
				//G.read_dat_file("C:\\Resources\\3D\\dat\\hazelnuts\\hnut_uint.dat");

				auto fn = [regularGrid, mesh](double x, double y, double z) -> double{
					//const double radius = 1.0;
					//const double cx = 2.0, cy = 2.0, cz = 2.0;
					//x -= cx; y -= cy; z -= cz;
					//return radius * radius - x * x - y * y - z * z;

					auto index = regularGrid->GetIndex(glm::vec3(x, y, z));
					auto cell = regularGrid->GetCell(index);
					if (cell)
					{
						for (auto& t : cell->GetTriangles())
						{
							auto v0 = mesh->GetVertex(t->v0->index);
							auto v1 = mesh->GetVertex(t->v1->index);
							auto v2 = mesh->GetVertex(t->v2->index);

							glm::vec3 centroid = (v0 + v1 + v2) / 3.0f;
							auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));

							auto p = glm::vec3(x, y, z);
							if (0 <= glm::dot(normal, p - centroid))
							{
								return 1.0;
							}
						}
					}
					else
					{
						return 0.0;
					}
					};

				auto aabbMin = regularGrid->GetMinPoint();
				auto aabbMax = regularGrid->GetMaxPoint();

				grid3d G;
				G.generate_grid_from_lambda(aabbMin.x, aabbMin.y, aabbMin.z, // coordinates of the grid origin
					aabbMax.x, aabbMax.y, aabbMax.z, // coordinates of the opposite corner
					cellSize, cellSize, cellSize, // steps
					fn);

				MC33 MC;
				MC.set_grid3d(G);

				surface S;
				MC.calculate_isosurface(S, 0.0f);

				auto not = S.get_num_triangles();
				cout << "not : " << not << endl;
				for (unsigned int i = 0; i < not; i++)
				{
					auto tis = S.getTriangle(i);
					auto v0 = S.getVertex(tis[0]);
					auto v1 = S.getVertex(tis[1]);
					auto v2 = S.getVertex(tis[2]);

					scene->Debug("MC33")->AddTriangle(
						glm::vec3(v0[0], v0[1], v0[2]), 
						glm::vec3(v1[0], v1[1], v1[2]), 
						glm::vec3(v2[0], v2[1], v2[2]), 
						glm::green, glm::green, glm::green);
				}
			}
			*/

			/*
			{
				int breakCount = 7;
				int count = 0;

				auto cells = regularGrid->GetCells();
				for (size_t z = 0; z < regularGrid->GetCellCountZ(); z++)
				{
					for (size_t y = 0; y < regularGrid->GetCellCountY(); y++)
					{
						for (size_t x = 0; x < regularGrid->GetCellCountX(); x++)
						{
							auto cell = regularGrid->GetCell(make_tuple(x, y, z));
							if (nullptr != cell)
							{
								if (0 < cell->GetTriangles().size())
								{
									count++;
									if (breakCount == count)
									{
										cout << "x : " << x << " , y : " << y << " , z : " << z << endl;

										scene->Debug("voxels")->AddBox(cell->GetCenter(), cell->GetXLength(), cell->GetYLength(), cell->GetZLength(), glm::blue);
									}
								}
							}
							if (breakCount == count) break;
						}
						if (breakCount == count) break;
					}
					if (breakCount == count) break;
				}
			}
			*/

			/*
			{
				auto cells = regularGrid->GetCells();
				auto cell = cells[1][0][1];

				struct GridCell
				{
					glm::vec3 vertex[8];
					float value[8];
				};

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
					//float distance = FLT_MAX;

					for (auto& t : cell->GetTriangles())
					{
						auto v0 = mesh->GetVertex(t->v0->index);
						auto v1 = mesh->GetVertex(t->v1->index);
						auto v2 = mesh->GetVertex(t->v2->index);


						if ((cell->Contains(v0) == false && cell->Contains(v1) == false && cell->Contains(v2) == false) || 
							(cell->Contains(v0) && cell->Contains(v1) == false && cell->Contains(v2) == false) ||
							(cell->Contains(v1) && cell->Contains(v2) == false && cell->Contains(v0) == false) ||
							(cell->Contains(v2) && cell->Contains(v0) == false && cell->Contains(v1) == false))
						{
							auto n = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
							auto c = (v0 + v1 + v2) / 3.0f;

							if (0 <= glm::dot(gridCell.vertex[i] - c, n))
							{
								gridCell.value[i] = -1.0f;
							}

							scene->Debug("faces")->AddTriangle(v0, v1, v2);
						}
					}
				}
			}
			*/

			auto result = regularGrid->ExtractSurface(0.0f);
			for (auto& vs : result)
			{
				scene->Debug("Result")->AddTriangle(vs[0], vs[1], vs[2], glm::vec4(0.7f, 0.6f, 0.4f, 1.0f), glm::vec4(0.7f, 0.6f, 0.4f, 1.0f), glm::vec4(0.7f, 0.6f, 0.4f, 1.0f));
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
