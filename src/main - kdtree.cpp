#include <iostream>

#include <Neon/Neon.h>

#include <Neon/CUDA/CUDAMesh.h>

#include <Neon/CUDA/CUDATest.h>
#include <Neon/CUDA/CUDATSDF.h>
#include <Neon/CUDA/CUDASurfaceExtraction.h>

int gindex = 0;

cudaGraphicsResource_t cuda_vbo_resource_test;

class HKDTreeNode
{
public:
	HKDTreeNode(const glm::vec3& p) : p(p) {}

	inline const glm::vec3& GetPosition() const { return p; }
	inline HKDTreeNode* GetLeft() const { return left; }
	inline void SetLeft(HKDTreeNode* node) { left = node; }
	inline HKDTreeNode* GetRight() const { return right; }
	inline void SetRight(HKDTreeNode* node) { right = node; }

	void* data = nullptr;

private:
	glm::vec3 p;
	HKDTreeNode* left = nullptr;
	HKDTreeNode* right = nullptr;
};

template<typename T>
class HKDTree
{
public:
	HKDTree() {}

	void Clear()
	{
		if (nullptr != root)
		{
			ClearRecursive(root);
			root = nullptr;
		}
	}

	void Insert(const glm::vec3& position)
	{
		root = InsertRecursive(root, position, 0);
	}

	HKDTreeNode* FindNearestNeighbor(HKDTreeNode* root, const glm::vec3& target, int depth)
	{
		nearestNeighbor = root->GetVertex();
		nearestNeighborDistance = glm::length(query - root->GetVertex()->p);
		FindNearestNeighborRecursive(root, query, 0);
		return nearestNeighbor;
	}

	HKDTreeNode* FindNearestNeighborNode(const glm::vec3& query)
	{
		if (nullptr == root)
			return nullptr;

		root->data = (void*)1;

		auto nearestNeighborNode = root;
		auto nearestNeighborDistance = distance(query, root->GetPosition());
		FindNearestNeighborRecursive(root, nearestNeighborNode, nearestNeighborDistance, query, 0);
		return nearestNeighborNode;
	}

	vector<T*> RangeSearch(const glm::vec3& query, float squaredRadius) const
	{
		vector<T*> result;
		RangeSearchRecursive(root, query, squaredRadius, result, 0);
		return result;
	}

	inline const HKDTreeNode* GetRoot() const { return root; }

	inline bool IsEmpty() const { return nullptr == root; }

private:
	HKDTreeNode* root = nullptr;
	HKDTreeNode* nearestNeighborNode = nullptr;

	void ClearRecursive(HKDTreeNode* node)
	{
		if (nullptr != node->GetLeft())
		{
			ClearRecursive(node->GetLeft());
		}

		if (nullptr != node->GetRight())
		{
			ClearRecursive(node->GetRight());
		}

		delete node;
	}

	HKDTreeNode* InsertRecursive(HKDTreeNode* node, const glm::vec3& position, int depth) {
		if (node == nullptr) {
			auto newNode = new HKDTreeNode(position);
			return newNode;
		}

		int currentDimension = depth % 3;
		if (((float*)&position)[currentDimension] < ((float*)&node->GetPosition())[currentDimension])
		{
			node->SetLeft(InsertRecursive(node->GetLeft(), position, depth + 1));
		}
		else {
			node->SetRight(InsertRecursive(node->GetRight(), position, depth + 1));
		}

		return node;
	}

	void FindNearestNeighborRecursive(HKDTreeNode* node, HKDTreeNode* nnNode, float nearestNeighborDistance, const glm::vec3& query, int depth) {
		if (node == nullptr) {
			return;
		}

		int currentDimension = depth % 3;
		float nnDistance = nearestNeighborDistance;

		auto nodeDistance = distance(query, node->GetPosition());
		if (nodeDistance < nnDistance) {
			nnNode = node;
			nnDistance = nodeDistance;
		}

		auto queryValue = ((float*)&query)[currentDimension];
		auto nodeValue = ((float*)&node->GetPosition())[currentDimension];

		HKDTreeNode* closerNode = (queryValue < nodeValue) ? node->GetLeft() : node->GetRight();
		HKDTreeNode* otherNode = (queryValue < nodeValue) ? node->GetRight() : node->GetLeft();

		FindNearestNeighborRecursive(closerNode, nnNode, nnDistance, query, depth + 1);

		// Check if the other subtree could have a closer point
		if (std::abs(queryValue - nodeValue) * std::abs(queryValue - nodeValue) < nnDistance) {
			FindNearestNeighborRecursive(otherNode, nnNode, nnDistance, query, depth + 1);
		}
	}

	void RangeSearchRecursive(HKDTreeNode* node, const glm::vec3& query, float squaredRadius, std::vector<T*>& result, int depth) const {
		if (node == nullptr) {
			return;
		}

		float nodeDistance = glm::length(query - node->GetVertex()->p);
		if (nodeDistance <= squaredRadius) {
			result.push_back(node->GetVertex());
		}

		int currentDimension = depth % 3;
		auto queryValue = ((float*)&query)[currentDimension];
		auto nodeValue = ((float*)&node->GetVertex()->p)[currentDimension];

		HKDTreeNode* closerNode = (queryValue < nodeValue) ? node->GetLeft() : node->GetRight();
		HKDTreeNode* otherNode = (queryValue < nodeValue) ? node->GetRight() : node->GetLeft();

		RangeSearchRecursive(closerNode, query, squaredRadius, result, depth + 1);

		// Check if the other subtree could have points within the range
		if (std::abs(queryValue - nodeValue) * std::abs(queryValue - nodeValue) <= squaredRadius) {
			RangeSearchRecursive(otherNode, query, squaredRadius, result, depth + 1);
		}
	}
};


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

#pragma region Draw Grid
		{
			size_t hResolution = 1000;
			size_t vResolution = 1000;
			float xUnit = 0.1f;
			float yUnit = 0.1f;

			float minX = -((float)hResolution * xUnit * 0.5f);
			float maxX = ((float)hResolution * xUnit * 0.5f);
			float minY = -((float)vResolution * yUnit * 0.5f);
			float maxY = ((float)vResolution * yUnit * 0.5f);

			for (float y = minY; y <= maxY; y += yUnit)
			{
				scene->Debug("grid lines")->AddLine({ minX, y, 0.0f }, { maxX, y, 0.0f }, glm::white, glm::white);
			}
			for (float x = minX; x <= maxX; x += xUnit)
			{
				scene->Debug("grid lines")->AddLine({ x, minY, 0.0f }, { x, maxY, 0.0f }, glm::white, glm::white);
			}
		}
		{
			size_t hResolution = 1000;
			size_t vResolution = 1000;
			float xUnit = 0.1f;
			float yUnit = 0.1f;

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

		auto kdtree = HKDTree<glm::vec3>();

		{
			auto entity = scene->CreateEntity("PLY");
			auto mesh = scene->CreateComponent<Neon::Mesh>("PLY Mesh");
			entity->AddComponent(mesh);

			mesh->FromXYZWFile("C:\\saveData\\Result\\globalVoxelValues.xyzw");

			for (auto& v : mesh->GetVertexBuffer()->GetElements())
			{
				//printf("%f, %f, %f\n", v.x, v.y, v.z);
				kdtree.Insert(v);
			}

			function<void(const HKDTreeNode*)> visualize;

			visualize = [&](const HKDTreeNode* node) {
				if (nullptr != node)
				{
					scene->Debug("KDTREE")->AddPoint(node->GetPosition());

					if (nullptr != node->GetLeft())
					{
						visualize(node->GetLeft());
					}
					if (nullptr != node->GetRight())
					{
						visualize(node->GetRight());
					}
				}
			};

			visualize(kdtree.GetRoot());

			for (auto& v : mesh->GetVertexBuffer()->GetElements())
			{
				auto node = kdtree.FindNearestNeighbor(v);
				if (nullptr == node)
				{
					scene->Debug("Error")->AddPoint(v, glm::red);
				}
				else
				{
					if (distance(v, node->GetPosition()) > 0.001f)
					{
						scene->Debug("Error")->AddPoint(v, glm::red);
					}
				}
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
