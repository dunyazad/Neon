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

		glPointSize(5);

		auto scene = app.CreateScene("Scene/Main");

		auto toggleVisibility = [](const Neon::KeyEvent& event)
		{
			auto entity = dynamic_cast<Neon::Entity*>(event.target);
			if (nullptr != entity)
			{
				auto mesh = entity->GetComponent<Neon::Mesh>(0);
				if (nullptr != mesh)
				{
					if (GLFW_KEY_ESCAPE == event.key && GLFW_RELEASE == event.action)
					{
						mesh->SetVisible(!mesh->IsVisible());
					}
				}
			}
		};

		auto toggleFillMode = [](const Neon::KeyEvent& event)
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

		auto debugPoints = scene->CreateDebugEntity("DebugEntity/Points");
		debugPoints->AddKeyEventHandler(toggleVisibility);
		debugPoints->AddKeyEventHandler(toggleFillMode);

		auto debugLines = scene->CreateDebugEntity("DebugEntity/Lines");
		debugLines->AddKeyEventHandler(toggleVisibility);
		debugLines->AddKeyEventHandler(toggleFillMode);

		auto debugTriangles = scene->CreateDebugEntity("DebugEntity/Triangles");
		debugTriangles->AddKeyEventHandler(toggleVisibility);
		debugTriangles->AddKeyEventHandler(toggleFillMode);

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

		{
			auto t = Neon::Time("Mesh Loading");

			auto entity = scene->CreateEntity("Entity/Mesh");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Mesh");
			entity->AddComponent(mesh);
			mesh->FromSTLFile(Neon::URL::Resource("/stl/mx.stl"), 0.01f, 0.01f, 0.01f);
			//auto nov = mesh->GetVertexBuffer()->Size();
			//auto rotation = glm::angleAxis(-glm::half_pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
			//for (int i = 0; i < nov; i++)
			//{
			//	mesh->AddColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

			//	{
			//		auto roatated = rotation * mesh->GetVertex(i);
			//		mesh->SetVertex(i, roatated);
			//	}

			//	//{
			//	//	float x, y, z;
			//	//	mesh->GetNormal(i, x, y, z);
			//	//	auto roatated = rotation * glm::vec3(x, y, z);
			//	//	mesh->SetNormal(i, roatated.x, roatated.y, roatated.z);
			//	//}
			//}

			mesh->FillColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

			mesh->RecalculateFaceNormal();

			auto noi = mesh->GetIndexBuffer()->Size();
			for (size_t i = 0; i < noi / 3; i++)
			{
				auto i0 = mesh->GetIndexBuffer()->GetElement(i * 3 + 0);
				auto i1 = mesh->GetIndexBuffer()->GetElement(i * 3 + 1);
				auto i2 = mesh->GetIndexBuffer()->GetElement(i * 3 + 2);

				auto v0 = mesh->GetVertex(i0);
				auto v1 = mesh->GetVertex(i1);
				auto v2 = mesh->GetVertex(i2);

				debugTriangles->AddTriangle(v0, v1, v2, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
			}

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			entity->AddComponent(shader);

			auto transform = scene->CreateComponent<Neon::Transform>("Transform/Mesh");
			entity->AddComponent(transform);
			entity->AddKeyEventHandler([mesh](const Neon::KeyEvent& event) {
				if (GLFW_KEY_1 == event.key && GLFW_RELEASE == event.action)
				{
					mesh->ToggleFillMode();
				}
				});

			auto bspTree = scene->CreateComponent<Neon::BSPTree<glm::vec3>>("BSPTree/Mesh", mesh);
			bspTree->Build();

			size_t count = 0;
			bspTree->Traverse(bspTree->root, [&count, debugTriangles](Neon::BSPTreeNode<glm::vec3>* node) {
				glPointSize(5.0f);

				count++;
				},
				[&count]() {
					cout << "total count : " << count << endl;
				});

			entity->AddMouseButtonEventHandler([scene, mesh, bspTree, debugPoints, debugLines](const Neon::MouseButtonEvent& event) {
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
						debugPoints->Clear();

						debugPoints->AddPoint(intersection, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

						auto result = bspTree->GetNearestNode(bspTree->root, intersection, bspTree->root);

						if (nullptr != result)
						{
							bspTree->Traverse(bspTree->root, [&mesh, result](Neon::BSPTreeNode<glm::vec3>* node) {
								//cout << node->index << endl;
								if (node->t < result->t)
								{
									mesh->SetColor(node->index, glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
								}
								}, []() {});

							//debugPoints->AddPoint(result->t, glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
						}

					}
				}
				});
		}
		});





	app.OnUpdate([&](double now, double timeDelta) {
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
