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
		
		{
			ifstream file("C:\\Resources\\Results\\Triangulated.bson", ios::binary);
			if (file.is_open())
			{
				file.seekg(0, ios::end);
				auto fileSize = file.tellg();
				file.seekg(0, ios::beg);

				vector<uint8_t> fileContents(static_cast<size_t>(fileSize));
				file.read(reinterpret_cast<char*>(fileContents.data()), fileSize);
				file.close();

				json json = json::from_bson(fileContents);

				vector<glm::vec3> vertices;
				vector<glm::vec3> normals;
				vector<glm::vec4> colors;
				vector<GLuint> indices;


				auto nov = json["number of vertices"].get<size_t>();
				if (0 < nov)
				{
					auto vs = json["vertices"].get<vector<float>>();
					for (size_t i = 0; i < nov; i++)
					{
						auto x = vs[i * 3 + 0];
						auto y = vs[i * 3 + 1];
						auto z = vs[i * 3 + 2];

						vertices.push_back({ x, y, z });
					}
				}

				auto non = json["number of normals"].get<size_t>();
				if (0 < non)
				{
					auto ns = json["normals"].get<vector<float>>();;
					for (size_t i = 0; i < non; i++)
					{
						auto x = ns[i * 3 + 0];
						auto y = ns[i * 3 + 1];
						auto z = ns[i * 3 + 2];

						normals.push_back({ x, y, z });
					}
				}

				auto noc = json["number of colors"].get<size_t>();
				if (0 < noc)
				{
					auto cs = json["colors"].get<vector<float>>();;
					for (size_t i = 0; i < noc; i++)
					{
						auto r = cs[i * 3 + 0];
						auto g = cs[i * 3 + 1];
						auto b = cs[i * 3 + 2];

						colors.push_back({ r, g, b, 1.0f });
					}
				}

				auto not = json["number of triangles"].get<size_t>();
				if (0 < not)
				{
					auto ts = json["triangles"].get<vector<tuple<unsigned int, unsigned int, unsigned int>>>();
					for (size_t i = 0; i < not; i++)
					{
						indices.push_back(get<0>(ts[i]));
						indices.push_back(get<1>(ts[i]));
						indices.push_back(get<2>(ts[i]));
					}

					for (size_t i = 0; i < not / 3; i++)
					{
						auto i0 = indices[i * 3 + 0];
						auto i1 = indices[i * 3 + 1];
						auto i2 = indices[i * 3 + 2];

						auto v0 = vertices[i0];
						auto v1 = vertices[i1];
						auto v2 = vertices[i2];

						auto c0 = colors[i0];
						auto c1 = colors[i1];
						auto c2 = colors[i2];

						scene->Debug("mesh")->AddTriangle(v0, v1, v2, c0, c1, c2);
					}
				}
			}
		}

	//	{
	//		int numberPoints = 60;
	//		
	//		std::default_random_engine eng(std::random_device{}());
	//		std::uniform_real_distribution<double> dist_w(0, 1280);
	//		std::uniform_real_distribution<double> dist_h(0, 1024);

	//		std::cout << "Generating " << numberPoints << " random points" << std::endl;

	//		vector<Eigen::Vector3f> input;

	//		std::vector<DT::Vector2<double>> points;
	//		for (int i = 0; i < numberPoints; ++i) {
	//			auto x = dist_w(eng);
	//			auto y = dist_h(eng);
	//			//printf("%f, %f\n", x, y);

	//			scene->Debug("points")->AddPoint(glm::vec3(x, y, 0.0f), glm::green);

	//			points.push_back(DT::Vector2<double>{x, y});

	//			input.push_back({ (float)x, (float)y, 0.0f });
	//		}
	//		
	//		auto result = NeonCUDA::DelaunayTriangulation_Custom(input);

	//		{
	//			float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
	//			float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;

	//			for (size_t i = 0; i < numberPoints; i++)
	//			{
	//				auto& v = input[i];

	//				if (minX > v.x()) minX = v.x();
	//				if (minY > v.y()) minY = v.y();
	//				if (minZ > v.z()) minZ = v.z();

	//				if (maxX < v.x()) maxX = v.x();
	//				if (maxY < v.y()) maxY = v.y();
	//				if (maxZ < v.z()) maxZ = v.z();
	//			}

	//			float centerX = (minX + maxX) * 0.5f;
	//			float centerY = (minY + maxY) * 0.5f;
	//			float centerZ = (minZ + maxZ) * 0.5f;

	//			input.push_back(Eigen::Vector3f(
	//				minX + (minX - centerX) * 3,
	//				minY + (minY - centerY) * 3,
	//				0.0f));

	//			input.push_back(Eigen::Vector3f(
	//				centerX,
	//				maxY + (maxY - centerY) * 3,
	//				0.0f));

	//			input.push_back(Eigen::Vector3f(
	//				maxX + (maxX - centerX) * 3,
	//				minY + (minY - centerY) * 3,
	//				0.0f));
	//		}

	//		{
	//			for (auto& t : result)
	//			{
	//				printf("%d, %d, %d\n", t.x(), t.y(), t.z());

	//				auto ix = t.x();
	//				auto iy = t.y();
	//				auto iz = t.z();

	//				if (ix == -1 || iy == -1 || iz == -1)
	//					continue;

	///*				if (ix >= input.size() - 3)
	//					continue;
	//				if (iy >= input.size() - 3)
	//					continue;
	//				if (iz >= input.size() - 3)
	//					continue;*/

	//				auto& v0 = input[ix];
	//				auto& v2 = input[iy];
	//				auto& v1 = input[iz];

	//				scene->Debug("Triangles")->AddTriangle(
	//					{ v0.x(), v0.y(), v0.z() },
	//					{ v1.x(), v1.y(), v1.z() },
	//					{ v2.x(), v2.y(), v2.z() });
	//			}
	//		}

	//		return;

	//		DT::Delaunay<double> triangulation;
	//		const auto start = std::chrono::high_resolution_clock::now();
	//		const std::vector<DT::Triangle<double>> triangles = triangulation.triangulate(points);
	//		const auto end = std::chrono::high_resolution_clock::now();
	//		const std::chrono::duration<double> diff = end - start;

	//		std::cout << triangles.size() << " triangles generated in " << diff.count() << "s\n";
	//		const std::vector<DT::Edge<double>> edges = triangulation.getEdges();

	//		for (auto& e : edges)
	//		{
	//			auto v0 = glm::vec3(e.v->x, e.v->y, 0.0f);
	//			auto v1 = glm::vec3(e.w->x, e.w->y, 0.0f);

	//			scene->Debug("lines")->AddLine(v0, v1, glm::red, glm::red);
	//		}

	//		triangulation.getTriangles();

	//		for (auto& t : triangles)
	//		{
	//			auto v0 = glm::vec3(t.a->x, t.a->y, 10.0f);
	//			auto v1 = glm::vec3(t.b->x, t.b->y, 10.0f);
	//			auto v2 = glm::vec3(t.c->x, t.c->y, 10.0f);

	//			scene->Debug("triangles")->AddTriangle(v0, v2, v1, glm::blue, glm::blue, glm::blue);
	//		}
	//	}
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
