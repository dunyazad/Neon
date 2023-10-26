#include <Neon/Component/NeonVETM.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/SpatialPartitioning/NeonSpatialHashing.h>

#include <Neon/NeonVertexBufferObject.hpp>

namespace Neon
{
	void VETM::KDTree::Clear()
	{
		if (nullptr != root)
		{
			ClearRecursive(root);
			root = nullptr;
		}
	}

	void VETM::KDTree::Insert(Vertex* vertex)
	{
		root = InsertRecursive(root, vertex, 0);
	}

	VETM::Vertex* VETM::KDTree::FindNearestNeighbor(const glm::vec3& query)
	{
		nearestNeighbor = root->GetVertex();
		nearestNeighborDistance = glm::length(query - root->GetVertex()->p);
		FindNearestNeighborRecursive(root, query, 0);
		return nearestNeighbor;
	}

	VETM::KDTreeNode* VETM::KDTree::FindNearestNeighborNode(const glm::vec3& query)
	{
		if (nullptr == root)
			return nullptr;

		nearestNeighborNode = root;
		nearestNeighbor = root->GetVertex();
		nearestNeighborDistance = glm::length(query - root->GetVertex()->p);
		FindNearestNeighborRecursive(root, query, 0);
		return nearestNeighborNode;
	}

	vector<VETM::Vertex*> VETM::KDTree::RangeSearch(const glm::vec3& query, float squaredRadius) const
	{
		vector<Vertex*> result;
		RangeSearchRecursive(root, query, squaredRadius, result, 0);
		return result;
	}

	void VETM::KDTree::ClearRecursive(KDTreeNode* node)
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

	VETM::KDTreeNode* VETM::KDTree::InsertRecursive(KDTreeNode* node, Vertex* vertex, int depth) {
		if (node == nullptr) {
			auto newNode = new KDTreeNode(vertex);
			return newNode;
		}

		int currentDimension = depth % 3;
		if (((float*)&vertex->p)[currentDimension] < ((float*)&node->GetVertex()->p)[currentDimension])
		{
			node->SetLeft(InsertRecursive(node->GetLeft(), vertex, depth + 1));
		}
		else {
			node->SetRight(InsertRecursive(node->GetRight(), vertex, depth + 1));
		}

		return node;
	}

	void VETM::KDTree::FindNearestNeighborRecursive(KDTreeNode* node, const glm::vec3& query, int depth) {
		if (node == nullptr) {
			return;
		}

		int currentDimension = depth % 3;

		auto nodeDistance = glm::length(query - node->GetVertex()->p);
		if (nodeDistance < nearestNeighborDistance) {
			nearestNeighborNode = node;
			nearestNeighbor = node->GetVertex();
			nearestNeighborDistance = nodeDistance;
		}

		auto queryValue = ((float*)&query)[currentDimension];
		auto nodeValue = ((float*)&node->GetVertex()->p)[currentDimension];

		KDTreeNode* closerNode = (queryValue < nodeValue) ? node->GetLeft() : node->GetRight();
		KDTreeNode* otherNode = (queryValue < nodeValue) ? node->GetRight() : node->GetLeft();

		FindNearestNeighborRecursive(closerNode, query, depth + 1);

		// Check if the other subtree could have a closer point
		if (std::abs(queryValue - nodeValue) * std::abs(queryValue - nodeValue) < nearestNeighborDistance) {
			FindNearestNeighborRecursive(otherNode, query, depth + 1);
		}
	}

	void VETM::KDTree::RangeSearchRecursive(KDTreeNode* node, const glm::vec3& query, float squaredRadius, std::vector<Vertex*>& result, int depth) const {
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

		KDTreeNode* closerNode = (queryValue < nodeValue) ? node->GetLeft() : node->GetRight();
		KDTreeNode* otherNode = (queryValue < nodeValue) ? node->GetRight() : node->GetLeft();

		RangeSearchRecursive(closerNode, query, squaredRadius, result, depth + 1);

		// Check if the other subtree could have points within the range
		if (std::abs(queryValue - nodeValue) * std::abs(queryValue - nodeValue) <= squaredRadius) {
			RangeSearchRecursive(otherNode, query, squaredRadius, result, depth + 1);
		}
	}

	VETM::VETM(const string& name, Mesh* mesh)
		: ComponentBase(name), mesh(mesh)
	{
	}

	VETM::~VETM()
	{
	}

	void VETM::Clear()
	{
		kdtree.Clear();

		for (auto vertex : vertices)
		{
			delete vertex;
		}
		vertices.clear();
		vid = 0;

		for (auto edge : edges)
		{
			delete edge;
		}
		edges.clear();
		eid = 0;
		edgeMapping.clear();

		for (auto triangle : triangles)
		{
			delete triangle;
		}
		triangles.clear();
		tid = 0;
		triangleMapping.clear();

		totalArea = 0.0;
	}

	void VETM::Clone(Mesh& clone)
	{
		clone.Clear();

		map<Vertex*, GLuint> vertexMapping;
		for (auto& v : vertices)
		{
			vertexMapping[v] = (GLuint)clone.AddVertex(v->p);
		}

		for (auto& t : triangles)
		{
			clone.AddTriangle(vertexMapping[t->v0], vertexMapping[t->v1], vertexMapping[t->v2]);
		}
	}

	void VETM::Build()
	{
		Clear();

		map<GLuint, Neon::VETM::Vertex*> vertexMapping;
		auto nov = mesh->GetVertexBuffer()->Size();
		for (size_t i = 0; i < nov; i++)
		{
			auto v = mesh->GetVertex(i);
			vertexMapping[i] = AddVertex(v, { 0.0f, 0.0f, 0.0f });
		}

		auto noi = mesh->GetIndexBuffer()->Size() / 3;
		for (size_t i = 0; i < noi; i++)
		{
			GLuint i0, i1, i2;
			mesh->GetTriangleVertexIndices(i, i0, i1, i2);

			auto v0 = vertexMapping[i0];
			auto v1 = vertexMapping[i1];
			auto v2 = vertexMapping[i2];

			AddTriangle(v0, v1, v2);
		}
	}

	VETM::Vertex* VETM::GetVertex(const glm::vec3& position)
	{
		auto nn = kdtree.FindNearestNeighborNode(position);
		if (nullptr != nn)
		{
			if (0.00001f > glm::length(nn->GetVertex()->p - position))
			{
				return nn->GetVertex();
			}
		}

		return nullptr;
	}

	VETM::Vertex* VETM::AddVertex(const glm::vec3& position, const glm::vec3& normal)
	{
		auto vertex = GetVertex(position);
		if (nullptr == vertex)
		{
			Vertex* vertex = new Vertex;
			vertex->id = vid++;
			vertex->p = position;
			vertex->n = normal;
			vertices.push_back(vertex);

			kdtree.Insert(vertex);

			return vertex;
		}
		else
		{
			return vertex;
		}
	}

	VETM::Edge* VETM::GetEdge(Vertex* v0, Vertex* v1)
	{
		auto t0 = make_tuple(v0, v1);
		auto t1 = make_tuple(v1, v0);
		if (edgeMapping.count(t0) != 0)
		{
			return edgeMapping[t0];
		}
		else if (edgeMapping.count(t1) != 0)
		{
			return edgeMapping[t1];
		}
		else
		{
			return nullptr;
		}
	}

	VETM::Edge* VETM::AddEdge(Vertex* v0, Vertex* v1)
	{
		auto edge = GetEdge(v0, v1);
		if (nullptr == edge)
		{
			edge = new Edge;
			edge->id = eid++;
			edge->v0 = v0;
			edge->v1 = v1;
			edge->length = glm::length(v0->p - v1->p);
			edges.insert(edge);
			v0->edges.insert(edge);
			v1->edges.insert(edge);
			edgeMapping[make_tuple(v0, v1)] = edge;
			edgeMapping[make_tuple(v1, v0)] = edge;
			return edge;
		}
		else
		{
			return edge;
		}
	}

	VETM::Vertex* VETM::GetCommonVertex(Edge* e0, Edge* e1)
	{
		auto e0v0 = e0->v0;
		auto e0v1 = e0->v1;
		auto e1v0 = e1->v0;
		auto e1v1 = e1->v1;

		if (e0v0 == e1v0)
			return e0v0;
		if (e0v0 == e1v1)
			return e0v0;
		if (e0v1 == e1v0)
			return e0v1;
		if (e0v1 == e1v1)
			return e0v1;

		return nullptr;
	}

	VETM::Triangle* VETM::GetTriangle(Vertex* v0, Vertex* v1, Vertex* v2)
	{
		auto e0 = GetEdge(v0, v1);
		auto e1 = GetEdge(v1, v2);
		auto e2 = GetEdge(v2, v0);

		auto t0 = make_tuple(e0, e1, e2);
		auto t1 = make_tuple(e1, e2, e0);
		auto t2 = make_tuple(e2, e0, e1);
		if (triangleMapping.count(t0) != 0)
		{
			return triangleMapping[t0];
		}
		else if (triangleMapping.count(t1) != 0)
		{
			return triangleMapping[t1];
		}
		else if (triangleMapping.count(t2) != 0)
		{
			return triangleMapping[t2];
		}
		else
		{
			return nullptr;
		}
	}

	VETM::Triangle* VETM::AddTriangle(Vertex* v0, Vertex* v1, Vertex* v2)
	{
		if (v0 == v1 || v1 == v2 || v2 == v0)
			return nullptr;

		auto triangle = GetTriangle(v0, v1, v2);
		if (nullptr == triangle)
		{
			triangle = new Triangle;
			triangle->id = tid++;

			auto e0 = AddEdge(v0, v1);
			auto e1 = AddEdge(v1, v2);
			auto e2 = AddEdge(v2, v0);

			triangle->v0 = v0;
			triangle->v1 = v1;
			triangle->v2 = v2;

			if (triangle->v0 == triangle->v1 || triangle->v1 == triangle->v2 || triangle->v2 == triangle->v0)
			{
				cout << "YYYYY" << endl;
			}
			if (triangle->v0 == triangle->v1 && triangle->v1 == triangle->v2)
			{
				cout << "XXXXX" << endl;
			}

			triangles.insert(triangle);
			triangleMapping[make_tuple(e0, e1, e2)] = triangle;
			triangleMapping[make_tuple(e1, e2, e0)] = triangle;
			triangleMapping[make_tuple(e2, e0, e1)] = triangle;

			v0->triangles.insert(triangle);
			v1->triangles.insert(triangle);
			v2->triangles.insert(triangle);

			e0->triangles.insert(triangle);
			e1->triangles.insert(triangle);
			e2->triangles.insert(triangle);

			triangle->centroid = { (triangle->v0->p.x + triangle->v1->p.x + triangle->v2->p.x) / 3,
									(triangle->v0->p.y + triangle->v1->p.y + triangle->v2->p.y) / 3,
									(triangle->v0->p.z + triangle->v1->p.z + triangle->v2->p.z) / 3 };

			auto d01 = (triangle->v1->p - triangle->v0->p);
			auto d02 = (triangle->v2->p - triangle->v0->p);
			auto area = glm::length(cross(d01, d02)) * float(0.5);
			totalArea += area;
			d01 = normalize(d01);
			d02 = normalize(d02);
			triangle->normal = normalize(cross(d01, d01));

			return triangle;
		}
		else
		{
			return triangle;
		}
	}

	void VETM::RemoveTriangle(Triangle* triangle)
	{
		auto v0 = triangle->v0;
		auto v1 = triangle->v1;
		auto v2 = triangle->v2;

		v0->triangles.erase(triangle);
		v1->triangles.erase(triangle);
		v2->triangles.erase(triangle);

		auto e0 = GetEdge(triangle->v0, triangle->v1);
		auto e1 = GetEdge(triangle->v1, triangle->v2);
		auto e2 = GetEdge(triangle->v2, triangle->v0);

		e0->triangles.erase(triangle);
		e1->triangles.erase(triangle);
		e2->triangles.erase(triangle);

		triangleMapping.erase(make_tuple(e0, e1, e2));
		triangleMapping.erase(make_tuple(e1, e2, e0));
		triangleMapping.erase(make_tuple(e2, e0, e1));
		triangles.erase(triangle);

		delete triangle;
	}

	set<VETM::Vertex*> VETM::GetAdjacentVertices(Vertex* vertex)
	{
		set<Vertex*> adjacentVertices;
		for (auto e : vertex->edges)
		{
			if (e->v0 != vertex)
			{
				adjacentVertices.insert(e->v0);
			}
			else if (e->v1 != vertex)
			{
				adjacentVertices.insert(e->v1);
			}
		}
		return adjacentVertices;
	}

	set<VETM::Vertex*> VETM::GetVerticesInRadius(const glm::vec3& position, float radius)
	{
		set<Vertex*> result;

		kdtree.RangeSearch(position, radius * radius);

		return result;
	}

	//float VETM::GetDistanceFromEdge(Edge* edge, const glm::vec3& position)
	//{
	//	auto ray = HRay(edge->v0->position, edge->v1->position - edge->v0->position);
	//	auto p = ray.GetNearestPointOnRay(position);
	//	return glm::distance(p, position);
	//}

	tuple<glm::vec3, glm::vec3, glm::vec3>
		VETM::GetTrianglePoints(Triangle* triangle)
	{
		auto p0 = triangle->v0->p;
		auto p1 = triangle->v1->p;
		auto p2 = triangle->v2->p;
		return make_tuple(p0, p1, p2);
	}

	bool compareByfloat(const std::tuple<float, VETM::Triangle*, glm::vec3>& tuple1,
		const std::tuple<float, VETM::Triangle*, glm::vec3>& tuple2)
	{
		return std::get<0>(tuple1) < std::get<0>(tuple2);
	}

	glm::vec3 VETM::GetTriangleCentroid(Triangle* triangle)
	{
		auto tps = GetTrianglePoints(triangle);
		auto& p0 = get<0>(tps);
		auto& p1 = get<1>(tps);
		auto& p2 = get<2>(tps);
		return { (p0.x + p1.x + p2.x) / 3, (p0.y + p1.y + p2.y) / 3,
				(p0.z + p1.z + p2.z) / 3 };
	}

	float VETM::GetTriangleArea(Triangle* triangle)
	{
		auto d01 = (triangle->v1->p - triangle->v0->p);
		auto d02 = (triangle->v2->p - triangle->v0->p);
		return glm::length(cross(d01, d02)) * float(0.5);
	}

	void VETM::FlipTriangle(Triangle* triangle)
	{
		auto t1 = triangle->v1;
		auto t2 = triangle->v2;
		triangle->v1 = t2;
		triangle->v2 = t1;

		triangle->normal = -triangle->normal;
	}

	//glm::vec3 VETM::GetNearestPointOnEdge(Edge* edge, const glm::vec3& position)
	//{
	//	auto ray = HRay(edge->v0->position, edge->v1->position - edge->v0->position);
	//	return ray.GetNearestPointOnRay(position);
	//}

	VETM::Vertex* VETM::GetNearestVertex(const glm::vec3& position)
	{
		auto node = kdtree.FindNearestNeighborNode(position);
		return node->GetVertex();
	}

	VETM::Vertex* VETM::GetNearestVertexOnTriangle(Triangle* triangle, const glm::vec3& position)
	{
		auto di0 = glm::length(position - triangle->v0->p);
		auto di1 = glm::length(position - triangle->v1->p);
		auto di2 = glm::length(position - triangle->v2->p);

		Vertex* nv = nullptr;
		if (di0 < di1 && di0 < di2)
		{
			nv = triangle->v0;
		}
		else if (di1 < di0 && di1 < di2)
		{
			nv = triangle->v1;
		}
		else if (di2 < di0 && di2 < di1)
		{
			nv = triangle->v2;
		}

		return nv;
	}

	//Edge* VETM::GetNearestEdgeOnTriangle(Triangle* triangle, const glm::vec3& position)
	//{
	//	//auto dp = point - origin;
	//	//auto distanceFromOrigin = direction * dp;
	//	//return origin + direction * distanceFromOrigin;

	//	auto d0 = GetDistanceFromEdge(triangle->e0, position);
	//	auto d1 = GetDistanceFromEdge(triangle->e1, position);
	//	auto d2 = GetDistanceFromEdge(triangle->e2, position);

	//	Edge* e = nullptr;
	//	if (d0 < d1 && d0 < d2)
	//	{
	//		e = triangle->e0;
	//	}
	//	else if (d1 < d0 && d1 < d2)
	//	{
	//		e = triangle->e1;
	//	}
	//	else if (d2 < d0 && d2 < d1)
	//	{
	//		e = triangle->e2;
	//	}
	//	return e;
	//}

	set<VETM::Triangle*> VETM::GetAdjacentTrianglesByEdge(Triangle* triangle)
	{
		set<Triangle*> adjacentTriangles;

		//adjacentTriangles.insert(triangle->e0->triangles.begin(),
		//	triangle->e0->triangles.end());
		//adjacentTriangles.insert(triangle->e1->triangles.begin(),
		//	triangle->e1->triangles.end());
		//adjacentTriangles.insert(triangle->e2->triangles.begin(),
		//	triangle->e2->triangles.end());

		adjacentTriangles.erase(triangle);

		return adjacentTriangles;
	}

	set<VETM::Triangle*> VETM::GetAdjacentTrianglesByVertex(Triangle* triangle)
	{
		set<Triangle*> adjacentTriangles;

		for (auto edge : triangle->v0->edges)
		{
			adjacentTriangles.insert(edge->triangles.begin(), edge->triangles.end());
		}

		for (auto edge : triangle->v1->edges)
		{
			adjacentTriangles.insert(edge->triangles.begin(), edge->triangles.end());
		}

		for (auto edge : triangle->v2->edges)
		{
			adjacentTriangles.insert(edge->triangles.begin(), edge->triangles.end());
		}

		adjacentTriangles.erase(triangle);

		return adjacentTriangles;
	}

	set<VETM::Triangle*> VETM::GetConnectedTriangles(Triangle* triangle)
	{
		set<Triangle*> visited;
		stack<Triangle*> triangleStack;
		triangleStack.push(triangle);
		while (triangleStack.empty() == false)
		{
			auto currentTriangle = triangleStack.top();
			triangleStack.pop();

			if (visited.count(currentTriangle) != 0)
			{
				continue;
			}
			visited.insert(currentTriangle);

			auto ats = GetAdjacentTrianglesByVertex(currentTriangle);
			for (auto at : ats)
			{
				if (visited.count(at) == 0)
				{
					triangleStack.push(at);
				}
			}
		}

		return visited;
	}

	vector<Mesh*> VETM::SeparateConnectedGroup()
	{
		vector<Mesh*> result;

		//set<Triangle*> visited;
		//for (auto triangle : triangles)
		//{
		//	if (visited.count(triangle) != 0)
		//	{
		//		continue;
		//	}

		//	auto group = GetConnectedTriangles(triangle);
		//	visited.insert(group.begin(), group.end());

		//	auto model = new Mesh(volume.GetVoxelSize());
		//	result.push_back(model);
		//	ExtractTriangles(*model, group);
		//}

		//sort(result.begin(), result.end(), [](Mesh* a, Mesh* b)
		//	{ return a->GetTotalArea() > b->GetTotalArea(); });

		return result;
	}

	vector<vector<VETM::Edge*>> VETM::GetBorderEdges()
	{
		vector<vector<Edge*>> result;

		set<Edge*> borderEdges;

		for (auto& edge : edges)
		{
			if (edge->triangles.size() < 2)
			{
				borderEdges.insert(edge);
			}
		}

		while (borderEdges.empty() == false)
		{
			vector<Edge*> border;
			Edge* seed = *borderEdges.begin();
			Edge* currentEdge = seed;
			set<Edge*> visited;
			do
			{
				if (0 != visited.count(currentEdge))
					break;

				visited.insert(currentEdge);
				border.push_back(currentEdge);
				borderEdges.erase(currentEdge);

				for (auto& ne : currentEdge->v1->edges)
				{
					if (ne->triangles.size() < 2)
					{
						if (ne->id != currentEdge->id)
						{
							//cout << "currentEdge->id : " << currentEdge->id << endl;

							currentEdge = ne;
							break;
						}
					}
				}
			} while (nullptr != currentEdge && currentEdge != seed);

			result.push_back(border);
		}
		return result;
	}

	void VETM::FillTrianglesToMakeBorderSmooth(float degreeMax)
	{
		while (true)
		{
			bool triangleAdded = false;
			auto foundBorderEdges = GetBorderEdges();

			for (size_t k = 0; k < foundBorderEdges.size(); k++)
			{
				auto borderEdges = foundBorderEdges[k];
				for (size_t i = 0; i < borderEdges.size() - 1; i += 2)
				{
					auto ce = borderEdges[i];
					auto ne = borderEdges[i + 1];
					auto cv = GetCommonVertex(ce, ne);
					VETM::Vertex* v0 = nullptr;
					VETM::Vertex* v1 = nullptr;
					VETM::Vertex* v2 = nullptr;
					if (ce->v0 == cv) v0 = ce->v1;
					else v0 = ce->v0;
					v1 = cv;
					if (ne->v0 == cv) v2 = ne->v1;
					else v2 = ne->v0;

					auto radian = angle(normalize(v0->p - v1->p), normalize(v2->p - v1->p));
					if (radian < degreeMax * PI / 180.0f)
					{
						auto n = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
						auto t = (*ce->triangles.begin());
						t->normal = normalize(cross(normalize(t->v1->p - t->v0->p), normalize(t->v2->p - t->v0->p)));
						if (dot(n, t->normal) < 0)
						{
							AddTriangle(v0, v2, v1);
							triangleAdded = true;
						}
					}
				}
			}

			if (false == triangleAdded)
				break;
		}
	}

	void VETM::ExtrudeBorder(const glm::vec3& direction, int segments)
	{
		auto foundBorderEdges = GetBorderEdges();

		for (size_t k = 0; k < foundBorderEdges.size(); k++)
		{
			vector<Edge*> borderEdges;
			vector<Edge*> newBorderEdges;

			for (size_t n = 0; n < segments; n++)
			{
				if (0 == n)
				{
					borderEdges = foundBorderEdges[k];
					newBorderEdges.resize(borderEdges.size());
				}
				for (size_t i = 0; i < borderEdges.size(); i++)
				{
					auto ce = borderEdges[i];
					auto ne = borderEdges[(i + 1) % borderEdges.size()];
					auto cv = GetCommonVertex(ce, ne);
					VETM::Vertex* v0 = nullptr;
					VETM::Vertex* v1 = nullptr;
					if (ce->v0 == cv) v0 = ce->v1;
					else v0 = ce->v0;
					v1 = cv;

					auto nv0 = AddVertex(v0->p + direction, { 0.0f, 0.0f, 0.0f });
					auto nv1 = AddVertex(v1->p + direction, { 0.0f, 0.0f, 0.0f });
					AddTriangle(v0, nv1, v1);
					AddTriangle(v0, nv0, nv1);

					newBorderEdges[i] = GetEdge(nv0, nv1);
				}
				swap(borderEdges, newBorderEdges);
				newBorderEdges.clear();
				newBorderEdges.resize(borderEdges.size());
			}
		}
	}

	/*void VETM::GenerateBase()
	{
		int segments = 20;
		int smoothOrder = 10;
		float stretch = 50.0f;

		auto foundBorderEdges = GetBorderEdges();

		glm::vec3 direction;
		for (size_t k = 0; k < foundBorderEdges.size(); k++)
		{
			vector<Edge*> borderEdges = foundBorderEdges[k];
			vector<Edge*> newBorderEdges;
			newBorderEdges.resize(borderEdges.size());
			vector<pair<float, float>> distances;

			AABB aabb;
#pragma region Determine direction
			for (size_t i = 0; i < borderEdges.size(); i++)
			{
				auto e = borderEdges[i];
				auto t = *e->triangles.begin();
				auto d = glm::vec3{ 0.0f, 0.0f, 0.0f, };
				aabb.Expand(e->v0->p);
				aabb.Expand(e->v1->p);

				if (t->v0 == e->v0 && t->v1 == e->v1 || t->v0 == e->v1 && t->v1 == e->v0)
				{
					d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v2->p);
				}
				else if (t->v1 == e->v0 && t->v2 == e->v1 || t->v2 == e->v1 && t->v1 == e->v0)
				{
					d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v0->p);
				}
				else if (t->v2 == e->v0 && t->v0 == e->v1 || t->v0 == e->v1 && t->v2 == e->v0)
				{
					d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v1->p);
				}

				direction += d;
			}

			direction = normalize(direction);
			if (abs(direction.x) > abs(direction.y) && abs(direction.x) > abs(direction.z))
			{
				if (0 < direction.x) direction = glm::vec3{ 1.0f, 0.0f, 0.0f };
				else direction = glm::vec3{ -1.0f, 0.0f, 0.0f };
			}
			else if (abs(direction.y) > abs(direction.x) && abs(direction.y) > abs(direction.z))
			{
				if (0 < direction.y) direction = glm::vec3{ 0.0f, 1.0f, 0.0f };
				else direction = glm::vec3{ 0.0f, -1.0f, 0.0f };
			}
			else if (abs(direction.z) > abs(direction.x) && abs(direction.z) > abs(direction.y))
			{
				if (0 < direction.z) direction = glm::vec3{ 0.0f, 0.0f, 1.0f };
				else direction = glm::vec3{ 0.0f, 0.0f, -1.0f };
			}

			for (size_t i = 0; i < borderEdges.size(); i++)
			{
				auto e = borderEdges[i];
				auto t = *e->triangles.begin();
				auto d = glm::vec3{ 0.0f, 0.0f, 0.0f, };
				if (abs(direction.x) > 0)
				{
					if (direction.x > 0)
					{
						distances.push_back({ aabb.GetMaxPoint().x - e->v0->p.x, aabb.GetMaxPoint().x - e->v1->p.x });
					}
					else
					{
						distances.push_back({ aabb.GetMinPoint().x - e->v0->p.x, aabb.GetMinPoint().x - e->v1->p.x });
					}
				}
				else if (abs(direction.y) > 0)
				{
					if (direction.y > 0)
					{
						distances.push_back({ aabb.GetMaxPoint().y - e->v0->p.y, aabb.GetMaxPoint().y - e->v1->p.y });
					}
					else
					{
						distances.push_back({ aabb.GetMinPoint().y - e->v0->p.y, aabb.GetMinPoint().y - e->v1->p.y });
					}
				}
				else if (abs(direction.z) > 0)
				{
					if (direction.z > 0)
					{
						distances.push_back({ aabb.GetMaxPoint().z - e->v0->p.z, aabb.GetMaxPoint().z - e->v1->p.z });
					}
					else
					{
						distances.push_back({ aabb.GetMinPoint().z - e->v0->p.z, aabb.GetMinPoint().z - e->v1->p.z });
					}
				}
			}

#pragma endregion

			vector<vector<Edge*>> newBorders;

			map<Vertex*, int> toSmooth;

			for (size_t n = 0; n < segments; n++)
			{
				for (size_t i = 0; i < borderEdges.size(); i++)
				{
					auto ce = borderEdges[i];
					auto ne = borderEdges[(i + 1) % borderEdges.size()];
					auto cv = GetCommonVertex(ce, ne);
					VETM::Vertex* v0 = nullptr;
					VETM::Vertex* v1 = nullptr;
					VETM::Vertex* v2 = nullptr;
					if (ce->v0 == cv) v0 = ce->v1;
					else v0 = ce->v0;
					v1 = cv;
					if (ne->v0 == cv) v2 = ne->v1;
					else v2 = ne->v0;

					if (n == 0)
					{
						toSmooth[v0] = 0;
						toSmooth[v1] = 0;
						toSmooth[v2] = 0;
					}

					auto d0 = (distances[i].first) / float(segments);
					auto d1 = (distances[i].second) / float(segments);

					d0 += stretch / float(segments);
					d1 += stretch / float(segments);

					auto p0 = v0->p + direction * d0;
					auto p1 = v1->p + direction * d1;

					auto nv0 = AddVertex(p0, { 0.0f, 0.0f, 0.0f });
					auto nv1 = AddVertex(p1, { 0.0f, 0.0f, 0.0f });
					AddTriangle(v0, nv1, v1);
					AddTriangle(v0, nv0, nv1);

					newBorderEdges[i] = GetEdge(nv0, nv1);
				}

				newBorders.push_back(newBorderEdges);

				swap(borderEdges, newBorderEdges);
				newBorderEdges.clear();
				newBorderEdges.resize(borderEdges.size());
			}

			for (int i = 0; i < smoothOrder; ++i)
			{
				map<Vertex*, int> temp;
				for (auto& kvp : toSmooth)
				{
					auto avs = GetAdjacentVertices(kvp.first);
					for (auto& nv : avs)
					{
						if (0 != toSmooth.count(nv))
						{
							temp[nv] = kvp.second + 1;
						}
						else
						{
							if (toSmooth[nv] > kvp.second + 1)
							{
								temp[nv] = kvp.second + 1;
							}
							else
							{
								temp[nv] = toSmooth[nv];
							}
						}
					}
				}
				swap(temp, toSmooth);
			}

			for (int k = 0; k < 10; ++k)
			{
				for (auto& nbs : newBorders)
				{
					for (size_t i = 0; i < nbs.size(); i++)
					{
						auto ce = nbs[i];
						auto ne = nbs[(i + 1) % nbs.size()];
						auto cv = GetCommonVertex(ce, ne);
						VETM::Vertex* v0 = nullptr;
						VETM::Vertex* v1 = nullptr;
						VETM::Vertex* v2 = nullptr;
						if (ce->v0 == cv) v0 = ce->v1;
						else v0 = ce->v0;
						v1 = cv;
						if (ne->v0 == cv) v2 = ne->v1;
						else v2 = ne->v0;

						v1->p = 0.5f * (v0->p + v1->p);
					}
				}

				for (auto& kvp : toSmooth)
				{
					glm::vec3 center;
					auto avs = GetAdjacentVertices(kvp.first);
					for (auto& nv : avs)
					{
						center += nv->p;
					}
					center /= avs.size();

					auto dir = center - kvp.first->p;
					auto dist = glm::length(dir);
					dir /= dist;
					kvp.first->p += dir * dist * ((float)kvp.second / (float)smoothOrder);
				}
			}
		}
	}*/

	void VETM::GenerateBase()
	{
		int segments = 20;
		int iteration = 10;
		int smoothOrder = 10;
		float stretch = 10.0f;

		auto foundBorderEdges = GetBorderEdges();

		for (size_t k = 0; k < foundBorderEdges.size(); k++)
		{
			glm::vec3 direction = glm::zero<glm::vec3>();

			vector<Edge*> borderEdges = foundBorderEdges[k];

			vector<Vertex*> borderVertices;

			AABB aabb;
#pragma region Determine direction
			{
				Neon::Time("Determine direction");
				for (size_t i = 0; i < borderEdges.size(); i++)
				{
					auto ce = borderEdges[i];
					auto ne = borderEdges[(i + 1) % borderEdges.size()];
					auto cv = GetCommonVertex(ce, ne);
					VETM::Vertex* v0 = nullptr;
					VETM::Vertex* v1 = nullptr;
					VETM::Vertex* v2 = nullptr;
					if (ce->v0 == cv) v0 = ce->v1;
					else v0 = ce->v0;
					v1 = cv;
					if (ne->v0 == cv) v2 = ne->v1;
					else v2 = ne->v0;

					borderVertices.push_back(v1);

					auto e = borderEdges[i];
					auto t = *e->triangles.begin();
					auto d = glm::vec3{ 0.0f, 0.0f, 0.0f, };
					aabb.Expand(e->v0->p);
					aabb.Expand(e->v1->p);

					if (t->v0 == e->v0 && t->v1 == e->v1 || t->v0 == e->v1 && t->v1 == e->v0)
					{
						d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v2->p);
					}
					else if (t->v1 == e->v0 && t->v2 == e->v1 || t->v2 == e->v1 && t->v1 == e->v0)
					{
						d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v0->p);
					}
					else if (t->v2 == e->v0 && t->v0 == e->v1 || t->v0 == e->v1 && t->v2 == e->v0)
					{
						d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v1->p);
					}

					direction += d;
				}
				direction /= borderEdges.size();

				direction = normalize(direction);
				if (abs(direction.x) > abs(direction.y) && abs(direction.x) > abs(direction.z))
				{
					if (0 < direction.x) direction = glm::vec3{ 1.0f, 0.0f, 0.0f };
					else direction = glm::vec3{ -1.0f, 0.0f, 0.0f };
				}
				else if (abs(direction.y) > abs(direction.x) && abs(direction.y) > abs(direction.z))
				{
					if (0 < direction.y) direction = glm::vec3{ 0.0f, 1.0f, 0.0f };
					else direction = glm::vec3{ 0.0f, -1.0f, 0.0f };
				}
				else if (abs(direction.z) > abs(direction.x) && abs(direction.z) > abs(direction.y))
				{
					if (0 < direction.z) direction = glm::vec3{ 0.0f, 0.0f, 1.0f };
					else direction = glm::vec3{ 0.0f, 0.0f, -1.0f };
				}
			}
#pragma endregion

#pragma region Border Smoothing
			{
				Neon::Time("Border Smoothing");

				for (size_t n = 0; n < iteration; n++)
				{
					for (size_t i = 0; i < borderVertices.size(); i++)
					{
						auto v0 = borderVertices[i];
						auto v1 = borderVertices[(i + 1) % borderVertices.size()];
						auto v2 = borderVertices[(i + 2) % borderVertices.size()];

						v1->p = 0.5f * (v0->p + v2->p);
					}
				}
			}
#pragma endregion

#pragma region Calculate distances
			vector<float> distances;
			{
				Neon::Time("Calculate distances");

				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v = borderVertices[i];

					if (abs(direction.x) > 0)
					{
						if (direction.x > 0)
						{
							distances.push_back(aabb.GetMaxPoint().x - v->p.x);
						}
						else
						{
							distances.push_back(aabb.GetMinPoint().x - v->p.x);
						}
					}
					else if (abs(direction.y) > 0)
					{
						if (direction.y > 0)
						{
							distances.push_back(aabb.GetMaxPoint().y - v->p.y);
						}
						else
						{
							distances.push_back(aabb.GetMinPoint().y - v->p.y);
						}
					}
					else if (abs(direction.z) > 0)
					{
						if (direction.z > 0)
						{
							distances.push_back(aabb.GetMaxPoint().z - v->p.z);
						}
						else
						{
							distances.push_back(aabb.GetMinPoint().z - v->p.z);
						}
					}
				}
			}
#pragma endregion

#pragma region Create Base Wall
			{
				Neon::Time("Create Base Wall");

				for (size_t n = 0; n < segments; n++)
				{
					vector<Vertex*> newVertices;
					for (size_t i = 0; i < borderVertices.size(); i++)
					{
						auto v0 = borderVertices[i];
						auto v1 = borderVertices[(i + 1) % borderVertices.size()];

						auto d0 = (distances[i]) / float(segments);
						auto d1 = (distances[(i + 1) % borderVertices.size()]) / float(segments);

						d0 += stretch / float(segments);
						d1 += stretch / float(segments);

						auto p0 = v0->p + direction * d0;
						auto p1 = v1->p + direction * d1;

						auto nv0 = AddVertex(p0, { 0.0f, 0.0f, 0.0f });
						auto nv1 = AddVertex(p1, { 0.0f, 0.0f, 0.0f });
						AddTriangle(v0, nv1, v1);
						AddTriangle(v0, nv0, nv1);

						newVertices.push_back(nv0);
					}
					swap(borderVertices, newVertices);
					newVertices.clear();
					newVertices.resize(borderVertices.size());
				}
			}
#pragma endregion

			/*
#pragma region Generate Floor
			using Point = std::array<float, 2>;
			if (abs(direction.x) > 0)
			{
				vector<Point> polygon;
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v = borderVertices[i];
					polygon.push_back({ v->p.y, v->p.z });
				}

				vector<vector<Point>> input;
				input.push_back(polygon);

				auto indices = mapbox::earcut<uint32_t>(input);

				for (size_t i = 0; i < indices.size() / 3; i++)
				{
					auto v0 = borderVertices[indices[i * 3 + 0]];
					auto v1 = borderVertices[indices[i * 3 + 1]];
					auto v2 = borderVertices[indices[i * 3 + 2]];

					auto normal = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
					if (dot(direction, normal) < 0)
					{
						AddTriangle(v0, v2, v1);
					}
					else
					{
						AddTriangle(v0, v1, v2);
					}
				}
			}
			else if (abs(direction.y) > 0)
			{
				vector<Point> polygon;
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v = borderVertices[i];
					polygon.push_back({ v->p.x, v->p.z });
				}

				vector<vector<Point>> input;
				input.push_back(polygon);

				auto indices = mapbox::earcut<uint32_t>(input);

				for (size_t i = 0; i < indices.size() / 3; i++)
				{
					auto v0 = borderVertices[indices[i * 3 + 0]];
					auto v1 = borderVertices[indices[i * 3 + 1]];
					auto v2 = borderVertices[indices[i * 3 + 2]];

					auto normal = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
					if (dot(direction, normal) < 0)
					{
						AddTriangle(v0, v2, v1);
					}
					else
					{
						AddTriangle(v0, v1, v2);
					}
				}
			}
			else if (abs(direction.z) > 0)
			{
				vector<Point> polygon;
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v = borderVertices[i];
					polygon.push_back({ v->p.x, v->p.y });
				}

				vector<vector<Point>> input;
				input.push_back(polygon);

				auto indices = mapbox::earcut<uint32_t>(input);

				for (size_t i = 0; i < indices.size() / 3; i++)
				{
					auto v0 = borderVertices[indices[i * 3 + 0]];
					auto v1 = borderVertices[indices[i * 3 + 1]];
					auto v2 = borderVertices[indices[i * 3 + 2]];

					auto normal = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
					if (dot(direction, normal) < 0)
					{
						AddTriangle(v0, v2, v1);
					}
					else
					{
						AddTriangle(v0, v1, v2);
					}
				}
			}
#pragma endregion
			*/
		}
	}

	void VETM::GenerateBaseWithHollow()
	{
		//TS(GenerateBaseWithHollow)

		int segments = 20;
		int iteration = 10;
		int smoothOrder = 10;
		float stretch = 10.0f;
		float hollowDistance = 3.0f;

		auto foundBorderEdges = GetBorderEdges();

		for (size_t k = 0; k < foundBorderEdges.size(); k++)
		{
			glm::vec3 direction;

			vector<Edge*> borderEdges = foundBorderEdges[k];

			vector<Vertex*> borderVertices;

			AABB aabb;
#pragma region Determine direction
			for (size_t i = 0; i < borderEdges.size(); i++)
			{
				auto ce = borderEdges[i];
				auto ne = borderEdges[(i + 1) % borderEdges.size()];
				auto cv = GetCommonVertex(ce, ne);
				VETM::Vertex* v0 = nullptr;
				VETM::Vertex* v1 = nullptr;
				VETM::Vertex* v2 = nullptr;
				if (ce->v0 == cv) v0 = ce->v1;
				else v0 = ce->v0;
				v1 = cv;
				if (ne->v0 == cv) v2 = ne->v1;
				else v2 = ne->v0;

				borderVertices.push_back(v1);

				auto e = borderEdges[i];
				auto t = *e->triangles.begin();
				auto d = glm::vec3{ 0.0f, 0.0f, 0.0f, };
				aabb.Expand(e->v0->p);
				aabb.Expand(e->v1->p);

				if (t->v0 == e->v0 && t->v1 == e->v1 || t->v0 == e->v1 && t->v1 == e->v0)
				{
					d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v2->p);
				}
				else if (t->v1 == e->v0 && t->v2 == e->v1 || t->v2 == e->v1 && t->v1 == e->v0)
				{
					d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v0->p);
				}
				else if (t->v2 == e->v0 && t->v0 == e->v1 || t->v0 == e->v1 && t->v2 == e->v0)
				{
					d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v1->p);
				}

				direction += d;
			}

			direction = normalize(direction);
			if (abs(direction.x) > abs(direction.y) && abs(direction.x) > abs(direction.z))
			{
				if (0 < direction.x) direction = glm::vec3{ 1.0f, 0.0f, 0.0f };
				else direction = glm::vec3{ -1.0f, 0.0f, 0.0f };
			}
			else if (abs(direction.y) > abs(direction.x) && abs(direction.y) > abs(direction.z))
			{
				if (0 < direction.y) direction = glm::vec3{ 0.0f, 1.0f, 0.0f };
				else direction = glm::vec3{ 0.0f, -1.0f, 0.0f };
			}
			else if (abs(direction.z) > abs(direction.x) && abs(direction.z) > abs(direction.y))
			{
				if (0 < direction.z) direction = glm::vec3{ 0.0f, 0.0f, 1.0f };
				else direction = glm::vec3{ 0.0f, 0.0f, -1.0f };
			}
#pragma endregion

#pragma region Border Smoothing
			for (size_t n = 0; n < iteration; n++)
			{
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v0 = borderVertices[i];
					auto v1 = borderVertices[(i + 1) % borderVertices.size()];
					auto v2 = borderVertices[(i + 2) % borderVertices.size()];

					v1->p = 0.5f * (v0->p + v2->p);
				}
			}
#pragma endregion

#pragma region Calculate distances
			vector<float> distances;
			for (size_t i = 0; i < borderVertices.size(); i++)
			{
				auto v = borderVertices[i];

				if (abs(direction.x) > 0)
				{
					if (direction.x > 0)
					{
						distances.push_back(aabb.GetMaxPoint().x - v->p.x);
					}
					else
					{
						distances.push_back(aabb.GetMinPoint().x - v->p.x);
					}
				}
				else if (abs(direction.y) > 0)
				{
					if (direction.y > 0)
					{
						distances.push_back(aabb.GetMaxPoint().y - v->p.y);
					}
					else
					{
						distances.push_back(aabb.GetMinPoint().y - v->p.y);
					}
				}
				else if (abs(direction.z) > 0)
				{
					if (direction.z > 0)
					{
						distances.push_back(aabb.GetMaxPoint().z - v->p.z);
					}
					else
					{
						distances.push_back(aabb.GetMinPoint().z - v->p.z);
					}
				}
			}
#pragma endregion

#pragma region Create Base Wall
			for (size_t n = 0; n < segments; n++)
			{
				vector<Vertex*> newVertices;
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v0 = borderVertices[i];
					auto v1 = borderVertices[(i + 1) % borderVertices.size()];

					auto d0 = (distances[i]) / float(segments);
					auto d1 = (distances[(i + 1) % borderVertices.size()]) / float(segments);

					d0 += stretch / float(segments);
					d1 += stretch / float(segments);

					auto p0 = v0->p + direction * d0;
					auto p1 = v1->p + direction * d1;

					auto nv0 = AddVertex(p0, { 0.0f, 0.0f, 0.0f });
					auto nv1 = AddVertex(p1, { 0.0f, 0.0f, 0.0f });
					AddTriangle(v0, nv1, v1);
					AddTriangle(v0, nv0, nv1);

					newVertices.push_back(nv0);
				}
				swap(borderVertices, newVertices);
				newVertices.clear();
				newVertices.resize(borderVertices.size());
			}
#pragma endregion

#pragma region Generate Floor
			/*
			using Point = std::array<float, 2>;
			if (abs(direction.x) > 0)
			{
				vector<Point> polygon;
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v = borderVertices[i];
					polygon.push_back({ v->p.y, v->p.z });
				}

				vector<vector<Point>> input;
				input.push_back(polygon);

				auto indices = mapbox::earcut<uint32_t>(input);

				for (size_t i = 0; i < indices.size() / 3; i++)
				{
					auto v0 = borderVertices[indices[i * 3 + 0]];
					auto v1 = borderVertices[indices[i * 3 + 1]];
					auto v2 = borderVertices[indices[i * 3 + 2]];

					auto normal = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
					if (dot(direction, normal) < 0)
					{
						AddTriangle(v0, v2, v1);
					}
					else
					{
						AddTriangle(v0, v1, v2);
					}
				}
			}
			else if (abs(direction.y) > 0)
			{
				vector<Point> polygon;
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v = borderVertices[i];
					polygon.push_back({ v->p.x, v->p.z });
				}

				vector<vector<Point>> input;
				input.push_back(polygon);

				auto indices = mapbox::earcut<uint32_t>(input);

				for (size_t i = 0; i < indices.size() / 3; i++)
				{
					auto v0 = borderVertices[indices[i * 3 + 0]];
					auto v1 = borderVertices[indices[i * 3 + 1]];
					auto v2 = borderVertices[indices[i * 3 + 2]];

					auto normal = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
					if (dot(direction, normal) < 0)
					{
						AddTriangle(v0, v2, v1);
					}
					else
					{
						AddTriangle(v0, v1, v2);
					}
				}
			}
			else if (abs(direction.z) > 0)
			{
				vector<Point> polygon;
				for (size_t i = 0; i < borderVertices.size(); i++)
				{
					auto v = borderVertices[i];
					polygon.push_back({ v->p.x, v->p.y });
				}

				vector<vector<Point>> input;
				input.push_back(polygon);

				auto indices = mapbox::earcut<uint32_t>(input);

				for (size_t i = 0; i < indices.size() / 3; i++)
				{
					auto v0 = borderVertices[indices[i * 3 + 0]];
					auto v1 = borderVertices[indices[i * 3 + 1]];
					auto v2 = borderVertices[indices[i * 3 + 2]];

					auto normal = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
					if (dot(direction, normal) < 0)
					{
						AddTriangle(v0, v2, v1);
					}
					else
					{
						AddTriangle(v0, v1, v2);
					}
				}
			}
			*/
#pragma endregion

#if 0
			for (size_t i = 0; i < 100; i++)
			{
#pragma region Calculate Face Normals
				TS(FaceNormal)
					for (auto& t : triangles)
					{
						t->normal = normalize(cross(normalize(t->v1->p - t->v0->p), normalize(t->v2->p - t->v0->p)));
					}
				TE(FaceNormal)
#pragma endregion

#pragma region Calculate Vertex Normals
					TS(VertexSmoothing)
					for (auto& v : vertices)
					{
						glm::vec3 normal;
						for (auto& t : v->triangles)
						{
							normal += t->normal;
						}
						normal /= v->triangles.size();
						normal = normalize(normal);

						glm::vec3 center;
						for (auto& e : v->edges)
						{
							if (e->v0 == v)
							{
								center += e->v1->p;
							}
							else if (e->v1 == v)
							{
								center += e->v0->p;
							}
						}
						center /= v->edges.size();

						v->p = center;

						v->p = (v->p - normal * hollowDistance / 100.0f) * 0.2f + center * 0.8f;
					}
				TE(VertexSmoothing)
#pragma endregion

					//#pragma region Mesh Shrink
					//				TS(MeshShrink)
					//				for (auto& v : vertices)
					//				{
					//					v->p = v->p - v->n * hollowDistance / 20.0f;
					//				}
					//				TE(MeshShrink)
					//#pragma endregion
			}
#endif // 0
		}

		//TE(GenerateBaseWithHollow)
	}

	bool lineTriangleIntersection(const glm::vec3& P1, const glm::vec3& P2, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, glm::vec3& intersection_point) {
		// Calculate line direction and normalize it
		glm::vec3 dir = P2 - P1;
		dir = normalize(dir);

		// Calculate the normal vector of the triangle's plane
		glm::vec3 N = cross(B - A, C - A);
		N = normalize(N);

		// Calculate the distance from the origin to the plane
		float d = -dot(N, A);

		// Calculate the parameter t of the intersection point
		float t = -(dot(N, P1) + d) / dot(N, dir);

		// Calculate the intersection point
		intersection_point = P1 + t * dir;

		// Calculate the barycentric coordinates of the intersection point
		float u, v, w;
		glm::vec3 v0 = C - A;
		glm::vec3 v1 = B - A;
		glm::vec3 v2 = intersection_point - A;

		float dot00 = dot(v0, v0);
		float dot01 = dot(v0, v1);
		float dot02 = dot(v0, v2);
		float dot11 = dot(v1, v1);
		float dot12 = dot(v1, v2);

		float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);

		u = (dot11 * dot02 - dot01 * dot12) * invDenom;
		v = (dot00 * dot12 - dot01 * dot02) * invDenom;
		w = 1.0f - u - v;

		// Check if the intersection point is inside the triangle
		if (u >= 0 && v >= 0 && u + v <= 1) {
			return true;
		}

		return false;
	}

	void VETM::DeleteSelfintersectingTriangles()
	{
		//TS(DeleteSelfintersectingTriangles)
		set<Triangle*> toDelete;

		for (auto& t0 : triangles)
		{
			if (0 != toDelete.count(t0))
				continue;

			bool intersects = false;
			for (auto& t1 : triangles)
			{
				if (0 != toDelete.count(t1))
					continue;

				glm::vec3 intersection;
				if (lineTriangleIntersection(t0->v0->p, t0->v1->p, t1->v0->p, t1->v2->p, t1->v2->p, intersection))
				{
					toDelete.insert(t0);
					toDelete.insert(t1);

					intersects = true;
					break;
				}
			}

			if (true == intersects)
				break;
		}

		for (auto& t : toDelete)
		{
			RemoveTriangle(t);
		}

		//TE(DeleteSelfintersectingTriangles)
	}

	void VETM::ApplyToMesh()
	{
		mesh->Clear();

		map<Vertex*, size_t> vertexMapping;
		for (auto& v : vertices)
		{
			vertexMapping[v] = mesh->AddVertex(v->p);
			mesh->AddColor(glm::white);
		}

		for (auto& t : triangles)
		{
			auto i0 = vertexMapping[t->v0];
			auto i1 = vertexMapping[t->v1];
			auto i2 = vertexMapping[t->v2];

			mesh->AddTriangle(i0, i1, i2);
		}

		mesh->RecalculateFaceNormal();
	}
}
