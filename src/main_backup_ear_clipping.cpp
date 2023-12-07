#include <iostream>

#include <Neon/Neon.h>

bool isEar(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const std::vector<glm::vec3>& polygon) {
	auto isConvex = [](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
		glm::vec2 ab = b - a;
		glm::vec2 bc = c - b;

		return 0.0f < ab.x * bc.y - ab.y * bc.x;
		};

	auto isPointInsideTriangle = [](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& point) {
		auto sign = [](const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3) {
			return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
			};

		bool d1 = sign(point, a, b) < 0.0f;
		bool d2 = sign(point, b, c) < 0.0f;
		bool d3 = sign(point, c, a) < 0.0f;

		return ((d1 == d2) && (d2 == d3));
		};

	if (!isConvex(a, b, c)) {
		return false;
	}

	for (const glm::vec3& point : polygon) {
		if (point != a && point != b && point != c) {
			if (isPointInsideTriangle(a, b, c, point)) {
				return false;
			}
		}
	}

	return true;
}

std::vector<int> earClipping(const std::vector<glm::vec3>& polygon) {
	std::vector<int> triangles;

	// Create a copy of the polygon to work with
	std::vector<glm::vec3> workingPolygon = polygon;

	while (workingPolygon.size() > 2) {
		bool earFound = false;

		for (size_t i = 0; i < workingPolygon.size(); ++i) {
			const glm::vec3& a = workingPolygon[(i + workingPolygon.size() - 1) % workingPolygon.size()];
			const glm::vec3& b = workingPolygon[i];
			const glm::vec3& c = workingPolygon[(i + 1) % workingPolygon.size()];

			// Check if the triangle (a, b, c) is a valid ear
			if (isEar(a, b, c, workingPolygon)) {
				// Find the indices of vertices in the original polygon
				int indexA = std::distance(polygon.begin(), std::find(polygon.begin(), polygon.end(), a));
				int indexB = std::distance(polygon.begin(), std::find(polygon.begin(), polygon.end(), b));
				int indexC = std::distance(polygon.begin(), std::find(polygon.begin(), polygon.end(), c));

				triangles.push_back(indexA);
				triangles.push_back(indexB);
				triangles.push_back(indexC);

				// Erase the middle vertex to "cut off" the ear
				workingPolygon.erase(workingPolygon.begin() + i);

				earFound = true;
				break;
			}
		}

		// If no ear is found, remove a vertex without checking
		if (!earFound) {
			workingPolygon.erase(workingPolygon.begin());
		}
	}

	// Add the last triangle
	for (const auto& vertex : workingPolygon) {
		int index = std::distance(polygon.begin(), std::find(polygon.begin(), polygon.end(), vertex));
		triangles.push_back(index);
	}

	return triangles;
}





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

		vector<glm::vec3> points;
		//{
		//	points.push_back(glm::vec3(3.10f, 33.8f, 0.0f));
		//	points.push_back(glm::vec3(10.5f, 29.8f, 0.0f));
		//	points.push_back(glm::vec3(3.4f, 21.9f, 0.0f));
		//	points.push_back(glm::vec3(8.0f, 3.3f, 0.0f));
		//	points.push_back(glm::vec3(9.1f, 20.1f, 0.0f));
		//	points.push_back(glm::vec3(14.1f, 16.0f, 0.0f));
		//	points.push_back(glm::vec3(10.9f, 15.7f, 0.0f));
		//	points.push_back(glm::vec3(10.2f, 4.9f, 0.0f));
		//	points.push_back(glm::vec3(17.3f, 17.0f, 0.0f));
		//	points.push_back(glm::vec3(20.3f, 3.3f, 0.0f));
		//	points.push_back(glm::vec3(36.5f, 5.8f, 0.0f));
		//	points.push_back(glm::vec3(24.2f, 7.3f, 0.0f));
		//	points.push_back(glm::vec3(37.7f, 14.2f, 0.0f));
		//	points.push_back(glm::vec3(37.9f, 30.4f, 0.0f));
		//	points.push_back(glm::vec3(33.8f, 16.4f, 0.0f));
		//	points.push_back(glm::vec3(24.8f, 12.3f, 0.0f));
		//	points.push_back(glm::vec3(26.4f, 36.8f, 0.0f));
		//}
		{
			points.push_back(glm::vec3(29.0f, 26.0f, 0.0f));
			points.push_back(glm::vec3(16.0f, 119.0f, 0.0f));
			points.push_back(glm::vec3(65.0f, 74.0f, 0.0f));
			points.push_back(glm::vec3(41.0f, 80.0f, 0.0f));
			points.push_back(glm::vec3(43.0f, 83.0f, 0.0f));
			points.push_back(glm::vec3(31.0f, 98.0f, 0.0f));
			points.push_back(glm::vec3(33.0f, 70.0f, 0.0f));
			points.push_back(glm::vec3(52.0f, 20.0f, 0.0f));
			points.push_back(glm::vec3(104.0f, 116.0f, 0.0f));
			points.push_back(glm::vec3(34.0f, 153.0f, 0.0f));
			points.push_back(glm::vec3(23.0f, 205.0f, 0.0f));
			points.push_back(glm::vec3(39.0f, 270.0f, 0.0f));
			points.push_back(glm::vec3(59.0f, 197.0f, 0.0f));
			points.push_back(glm::vec3(38.0f, 219.0f, 0.0f));
			points.push_back(glm::vec3(34.0f, 192.0f, 0.0f));
			points.push_back(glm::vec3(47.0f, 161.0f, 0.0f));
			points.push_back(glm::vec3(112.0f, 187.0f, 0.0f));
			points.push_back(glm::vec3(75.0f, 276.0f, 0.0f));
			points.push_back(glm::vec3(24.0f, 302.0f, 0.0f));
			points.push_back(glm::vec3(25.0f, 360.0f, 0.0f));
			points.push_back(glm::vec3(42.0f, 376.0f, 0.0f));
			points.push_back(glm::vec3(68.0f, 359.0f, 0.0f));
			points.push_back(glm::vec3(49.0f, 345.0f, 0.0f));
			points.push_back(glm::vec3(50.0f, 313.0f, 0.0f));
			points.push_back(glm::vec3(122.0f, 289.0f, 0.0f));
			points.push_back(glm::vec3(136.0f, 329.0f, 0.0f));
			points.push_back(glm::vec3(119.0f, 363.0f, 0.0f));
			points.push_back(glm::vec3(93.0f, 325.0f, 0.0f));
			points.push_back(glm::vec3(71.0f, 321.0f, 0.0f));
			points.push_back(glm::vec3(79.0f, 357.0f, 0.0f));
			points.push_back(glm::vec3(97.0f, 382.0f, 0.0f));
			points.push_back(glm::vec3(147.0f, 383.0f, 0.0f));
			points.push_back(glm::vec3(180.0f, 374.0f, 0.0f));
			points.push_back(glm::vec3(342.0f, 379.0f, 0.0f));
			points.push_back(glm::vec3(383.0f, 357.0f, 0.0f));
			points.push_back(glm::vec3(377.0f, 193.0f, 0.0f));
			points.push_back(glm::vec3(379.0f, 24.0f, 0.0f));
			points.push_back(glm::vec3(331.0f, 18.0f, 0.0f));
			points.push_back(glm::vec3(354.0f, 320.0f, 0.0f));
			points.push_back(glm::vec3(258.0f, 328.0f, 0.0f));
			points.push_back(glm::vec3(179.0f, 343.0f, 0.0f));
			points.push_back(glm::vec3(190.0f, 256.0f, 0.0f));
			points.push_back(glm::vec3(248.0f, 299.0f, 0.0f));
			points.push_back(glm::vec3(320.0f, 302.0f, 0.0f));
			points.push_back(glm::vec3(336.0f, 199.0f, 0.0f));
			points.push_back(glm::vec3(324.0f, 38.0f, 0.0f));
			points.push_back(glm::vec3(299.0f, 19.0f, 0.0f));
			points.push_back(glm::vec3(252.0f, 11.0f, 0.0f));
			points.push_back(glm::vec3(181.0f, 11.0f, 0.0f));
			points.push_back(glm::vec3(140.0f, 14.0f, 0.0f));
			points.push_back(glm::vec3(117.0f, 22.0f, 0.0f));
			points.push_back(glm::vec3(138.0f, 81.0f, 0.0f));
			points.push_back(glm::vec3(151.0f, 146.0f, 0.0f));
			points.push_back(glm::vec3(159.0f, 191.0f, 0.0f));
			points.push_back(glm::vec3(189.0f, 179.0f, 0.0f));
			points.push_back(glm::vec3(175.0f, 127.0f, 0.0f));
			points.push_back(glm::vec3(163.0f, 64.0f, 0.0f));
			points.push_back(glm::vec3(179.0f, 38.0f, 0.0f));
			points.push_back(glm::vec3(275.0f, 41.0f, 0.0f));
			points.push_back(glm::vec3(307.0f, 49.0f, 0.0f));
			points.push_back(glm::vec3(277.0f, 80.0f, 0.0f));
			points.push_back(glm::vec3(240.0f, 58.0f, 0.0f));
			points.push_back(glm::vec3(207.0f, 59.0f, 0.0f));
			points.push_back(glm::vec3(201.0f, 109.0f, 0.0f));
			points.push_back(glm::vec3(215.0f, 136.0f, 0.0f));
			points.push_back(glm::vec3(246.0f, 156.0f, 0.0f));
			points.push_back(glm::vec3(260.0f, 133.0f, 0.0f));
			points.push_back(glm::vec3(250.0f, 97.0f, 0.0f));
			points.push_back(glm::vec3(274.0f, 89.0f, 0.0f));
			points.push_back(glm::vec3(300.0f, 103.0f, 0.0f));
			points.push_back(glm::vec3(310.0f, 147.0f, 0.0f));
			points.push_back(glm::vec3(301.0f, 177.0f, 0.0f));
			points.push_back(glm::vec3(280.0f, 136.0f, 0.0f));
			points.push_back(glm::vec3(286.0f, 110.0f, 0.0f));
			points.push_back(glm::vec3(266.0f, 106.0f, 0.0f));
			points.push_back(glm::vec3(263.0f, 160.0f, 0.0f));
			points.push_back(glm::vec3(236.0f, 176.0f, 0.0f));
			points.push_back(glm::vec3(208.0f, 149.0f, 0.0f));
			points.push_back(glm::vec3(205.0f, 202.0f, 0.0f));
			points.push_back(glm::vec3(215.0f, 234.0f, 0.0f));
			points.push_back(glm::vec3(247.0f, 232.0f, 0.0f));
			points.push_back(glm::vec3(268.0f, 190.0f, 0.0f));
			points.push_back(glm::vec3(273.0f, 165.0f, 0.0f));
			points.push_back(glm::vec3(289.0f, 183.0f, 0.0f));
			points.push_back(glm::vec3(315.0f, 183.0f, 0.0f));
			points.push_back(glm::vec3(323.0f, 169.0f, 0.0f));
			points.push_back(glm::vec3(323.0f, 218.0f, 0.0f));
			points.push_back(glm::vec3(317.0f, 270.0f, 0.0f));
			points.push_back(glm::vec3(305.0f, 211.0f, 0.0f));
			points.push_back(glm::vec3(301.0f, 282.0f, 0.0f));
			points.push_back(glm::vec3(288.0f, 208.0f, 0.0f));
			points.push_back(glm::vec3(283.0f, 277.0f, 0.0f));
			points.push_back(glm::vec3(269.0f, 248.0f, 0.0f));
			points.push_back(glm::vec3(236.0f, 253.0f, 0.0f));
			points.push_back(glm::vec3(261.0f, 260.0f, 0.0f));
			points.push_back(glm::vec3(282.0f, 291.0f, 0.0f));
			points.push_back(glm::vec3(192.0f, 237.0f, 0.0f));
			points.push_back(glm::vec3(125.0f, 186.0f, 0.0f));
			points.push_back(glm::vec3(117.0f, 96.0f, 0.0f));
			points.push_back(glm::vec3(101.0f, 10.0f, 0.0f));
			points.push_back(glm::vec3(32.0f, 11.0f, 0.0f));
		}

		reverse(points.begin(), points.end());

		for (size_t i = 0; i < points.size(); i++)
		{
			auto pc = points[i];
			auto pn = points[(i + 1) % points.size()];

			scene->Debug("Points")->AddPoint(pc, glm::blue);

			scene->Debug("Edges")->AddLine(pc, pn, glm::red, glm::red);
		}

		int index = 0;
		scene->Debug("Boxes")->AddBox(points[0], 5, 5, 5, glm::red);
		scene->Debug("Boxes")->AddBox(points[1], 5, 5, 5, glm::green);
		scene->Debug("Boxes")->AddBox(points[2], 5, 5, 5, glm::blue);

		auto triangles = earClipping(points);
		for (size_t i = 0; i < triangles.size() / 3; i++)
		{
			auto i0 = triangles[i * 3 + 0];
			auto i1 = triangles[i * 3 + 1];
			auto i2 = triangles[i * 3 + 2];

			auto p0 = points[i0];
			auto p1 = points[i1];
			auto p2 = points[i2];

			scene->Debug("Triangles")->AddTriangle(p0, p1, p2);
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
