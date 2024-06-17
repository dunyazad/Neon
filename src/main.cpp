#include <iostream>

#include <Neon/Neon.h>

#include <Neon/CUDA/CUDAMesh.h>

#include <Neon/CUDA/CUDATest.h>
#include <Neon/CUDA/CUDATSDF.h>
#include <Neon/CUDA/CUDASurfaceExtraction.h>

#include <Neon/CUDA/cuOctree.h>


struct Point {
	double x, y, z;
	int clusterId = 0;  // 0 means unclassified, -1 means noise
};

double distance(const Point& p1, const Point& p2) {
	return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}

std::vector<Point> generate_3d_points(int num_points, int clusters) {
	std::vector<Point> points;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-10, 10);
	std::normal_distribution<> d(0, 1);

	for (int i = 0; i < clusters; ++i) {
		Point center = { dis(gen), dis(gen), dis(gen) };
		for (int j = 0; j < num_points / clusters; ++j) {
			points.push_back({ center.x + d(gen), center.y + d(gen), center.z + d(gen) });
		}
	}
	return points;
}

std::vector<int> region_query(const std::vector<Point>& points, int idx, double eps) {
	std::vector<int> neighbors;
	for (int i = 0; i < points.size(); ++i) {
		if (distance(points[idx], points[i]) < eps) {
			neighbors.push_back(i);
		}
	}
	return neighbors;
}

void expand_cluster(std::vector<Point>& points, int idx, std::vector<int>& neighbors,
	std::set<int>& visited, double eps, int min_pts, int cluster_id) {
	points[idx].clusterId = cluster_id;
	while (!neighbors.empty()) {
		int neighbor_idx = neighbors.back();
		neighbors.pop_back();
		if (visited.find(neighbor_idx) == visited.end()) {
			visited.insert(neighbor_idx);
			std::vector<int> new_neighbors = region_query(points, neighbor_idx, eps);
			if (new_neighbors.size() >= min_pts) {
				neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
			}
		}
		if (points[neighbor_idx].clusterId == 0) {
			points[neighbor_idx].clusterId = cluster_id;
		}
	}
}

void dbscan(std::vector<Point>& points, double eps, int min_pts) {
	std::set<int> visited;
	int cluster_id = 0;

	for (int i = 0; i < points.size(); ++i) {
		if (visited.find(i) != visited.end()) {
			continue;
		}
		visited.insert(i);

		std::vector<int> neighbors = region_query(points, i, eps);
		if (neighbors.size() < min_pts) {
			points[i].clusterId = -1;  // Mark as noise
		}
		else {
			cluster_id++;
			expand_cluster(points, i, neighbors, visited, eps, min_pts, cluster_id);
		}
	}
}

void print_clusters(const std::vector<Point>& points) {
	int max_cluster_id = 0;
	for (const auto& point : points) {
		if (point.clusterId > max_cluster_id) {
			max_cluster_id = point.clusterId;
		}
	}

	for (int cluster_id = 1; cluster_id <= max_cluster_id; ++cluster_id) {
		std::cout << "Cluster " << cluster_id << ":\n";
		for (int i = 0; i < points.size(); ++i) {
			if (points[i].clusterId == cluster_id) {
				std::cout << "(" << points[i].x << ", " << points[i].y << ", " << points[i].z << ") [Index: " << i << "]\n";
			}
		}
		std::cout << std::endl;
	}

	std::cout << "Noise:\n";
	for (int i = 0; i < points.size(); ++i) {
		if (points[i].clusterId == -1) {
			std::cout << "(" << points[i].x << ", " << points[i].y << ", " << points[i].z << ") [Index: " << i << "]\n";
		}
	}
	std::cout << std::endl;
}


int main()
{
	Neon::Application app(1280, 1024);
	Neon::URL::ChangeDirectory("..");

	//app.GetWindow()->UseVSync(false);
	//app.GetWindow()->SetWindowPosition(4920, 32);

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
				else if ((GLFW_KEY_GRAVE_ACCENT == event.key) && GLFW_RELEASE == event.action)
				{
					auto debugEntities = scene->GetDebugEntities();
					for (auto& entity : debugEntities)
					{
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
				else if ((GLFW_KEY_ENTER == event.key) && (GLFW_RELEASE == event.action || GLFW_REPEAT == event.action)) {
					static int pressCount = -1;
					pressCount++;

					int cnt = 0;
					int cx = 0;
					int cy = 0;
					int cz = 0;
					int offset = 3;

					int currentOffset = 0;

					while (currentOffset <= offset)
					{
						for (int z = -currentOffset; z <= currentOffset; z++)
						{
							for (int y = -currentOffset; y <= currentOffset; y++)
							{
								for (int x = -currentOffset; x <= currentOffset; x++)
								{
									if ((x == -currentOffset || x == currentOffset) ||
										(y == -currentOffset || y == currentOffset) ||
										(z == -currentOffset || z == currentOffset))
									{
										if (cnt == pressCount)
										{
											printf("[%2d] %d, %d, %d\n", cnt++, x, y, z);

											scene->Debug("cubes")->AddBox({ x, y, z }, 0.5f, 0.5f, 0.5f);
											return;
										}

										cnt++;
									}
								}
							}
						}

						currentOffset++;
					}
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

#pragma region Draw Grid
		//{
		//	size_t hResolution = 1000;
		//	size_t vResolution = 1000;
		//	float xUnit = 0.1f;
		//	float yUnit = 0.1f;

		//	float minX = -((float)hResolution * xUnit * 0.5f);
		//	float maxX = ((float)hResolution * xUnit * 0.5f);
		//	float minY = -((float)vResolution * yUnit * 0.5f);
		//	float maxY = ((float)vResolution * yUnit * 0.5f);

		//	for (float y = minY; y <= maxY; y += yUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ minX, y, 0.0f }, { maxX, y, 0.0f }, glm::white, glm::white);
		//	}
		//	for (float x = minX; x <= maxX; x += xUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ x, minY, 0.0f }, { x, maxY, 0.0f }, glm::white, glm::white);
		//	}
		//}
		//{
		//	size_t hResolution = 1000;
		//	size_t vResolution = 1000;
		//	float xUnit = 0.1f;
		//	float yUnit = 0.1f;

		//	float minX = -((float)hResolution * xUnit * 0.5f) + xUnit * 0.5f;
		//	float maxX = ((float)hResolution * xUnit * 0.5f) - xUnit * 0.5f;
		//	float minY = -((float)vResolution * yUnit * 0.5f) + yUnit * 0.5f;
		//	float maxY = ((float)vResolution * yUnit * 0.5f) - yUnit * 0.5f;

		//	for (float y = minY; y <= maxY; y += yUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ minX, y, 0.0f }, { maxX, y, 0.0f }, glm::gray, glm::gray);
		//	}
		//	for (float x = minX; x <= maxX; x += xUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ x, minY, 0.0f }, { x, maxY, 0.0f }, glm::gray, glm::gray);
		//	}
		//}
#pragma endregion

		{
			//	auto entity = scene->CreateEntity("Entity/spot");
			//	auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/spot");
			//	entity->AddComponent(mesh);

			//	mesh->FromSTLFile(Neon::URL::Resource("/stl/mesh.stl"));
			//	mesh->FillColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
			//	mesh->RecalculateFaceNormal();

			//	scene->GetMainCamera()->centerPosition = mesh->GetAABB().GetCenter();
			//	scene->GetMainCamera()->distance = mesh->GetAABB().GetDiagonalLength();

			//	auto shader = scene->CreateComponent<Neon::Shader>("Shader/Lighting", Neon::URL::Resource("/shader/lighting.vs"), Neon::URL::Resource("/shader/lighting.fs"));
			//	entity->AddComponent(shader);

			//	auto transform = scene->CreateComponent<Neon::Transform>("Transform/spot");
			//	entity->AddComponent(transform);
			//	entity->AddKeyEventHandler([mesh](const Neon::KeyEvent& event) {
			//		if (GLFW_KEY_ESCAPE == event.key && GLFW_RELEASE == event.action)
			//		{
			//			mesh->ToggleFillMode();
			//		}
			//		else if (GLFW_KEY_KP_ADD == event.key && (GLFW_PRESS == event.action || GLFW_REPEAT == event.action))
			//		{
			//			GLfloat currentPointSize;
			//			glGetFloatv(GL_POINT_SIZE, &currentPointSize);
			//			glPointSize(currentPointSize + 1.0f);
			//		}
			//		else if (GLFW_KEY_KP_SUBTRACT == event.key && (GLFW_PRESS == event.action || GLFW_REPEAT == event.action))
			//		{
			//			GLfloat currentPointSize;
			//			glGetFloatv(GL_POINT_SIZE, &currentPointSize);
			//			glPointSize(currentPointSize - 1.0f);
			//		}
			//		else if (GLFW_KEY_S == event.key && GLFW_RELEASE == event.action)
			//		{
			//			mesh->ToSTLFile("C:\\Users\\USER\\Desktop\\result.stl");
			//		}
			//		});

			//	entity->AddMouseButtonEventHandler([entity, scene, mesh](const Neon::MouseButtonEvent& event) {
			//		if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_DOUBLE_ACTION)
			//		{
			//			auto camera = scene->GetMainCamera();

			//			auto ray = camera->GetPickingRay(event.xpos, event.ypos);

			//			glm::vec3 intersection;
			//			size_t faceIndex = 0;
			//			if (mesh->Pick(ray, intersection, faceIndex))
			//			{
			//				camera->centerPosition = intersection;
			//			}
			//		}
			//		else if (event.button == GLFW_MOUSE_BUTTON_1 && event.action == GLFW_RELEASE)
			//		{
			//			auto camera = scene->GetMainCamera();

			//			auto ray = camera->GetPickingRay(event.xpos, event.ypos);

			//			glm::vec3 intersection;
			//			size_t faceIndex = 0;
			//			if (mesh->Pick(ray, intersection, faceIndex))
			//			{
			//				//debugPoints->Clear();

			//				//debugPoints->AddPoint(intersection, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

			//				auto vetm = entity->GetComponent<Neon::VETM>();
			//				auto vertex = vetm->GetNearestVertex(intersection);

			//				scene->Debug("Points")->Clear();
			//				scene->Debug("Points")->AddPoint(vertex->p, glm::red);
			//				scene->Debug("Points")->AddPoint(intersection, glm::white);

			//				scene->Debug("Lines")->Clear();
			//				cout << "---------------------------------------" << endl;
			//				for (auto& e : vertex->edges)
			//				{
			//					if (2 > e->triangles.size())
			//					{
			//						scene->Debug("Lines")->AddLine(e->v0->p, e->v1->p, glm::green, glm::green);
			//						cout << e->id << " : " << e->triangles.size() << endl;
			//					}
			//				}
			//			}
			//		}
			//		});
		}

		{
			int cx = 0;
			int cy = 0;
			int cz = 0;
			int offset = 3;

			int currentOffset = 0;

			int cnt = 0;
			while (currentOffset <= offset)
			{
				for (int z = -currentOffset; z <= currentOffset; z++)
				{
					for (int y = -currentOffset; y <= currentOffset; y++)
					{
						for (int x = -currentOffset; x <= currentOffset; x++)
						{
							if ((x == -currentOffset || x == currentOffset) ||
								(y == -currentOffset || y == currentOffset) ||
								(z == -currentOffset || z == currentOffset))
							{
								//printf("%d, %d, %d,\n", cx + x, cy + y, cz + z);

								scene->Debug("cubes")->AddBox({ cx + x, cy + y, cz + z }, 0.5f, 0.5f, 0.5f);
							}
						}
					}
				}

				currentOffset++;
			}
		}

		RunOctreeExample();

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
