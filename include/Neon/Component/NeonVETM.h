#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	class VETM : public ComponentBase
	{
	public:
		struct Vertex;
		struct Edge;
		struct Triangle;

		struct Vertex
		{
			int id = -1;
			glm::vec3 p = glm::zero<glm::vec3>();
			glm::vec3 n = glm::zero<glm::vec3>();
			set<Edge*> edges;
			set<Triangle*> triangles;
		};

		struct Edge
		{
			int id = -1;
			Vertex* v0 = nullptr;
			Vertex* v1 = nullptr;
			set<Triangle*> triangles;
			float length = 0.0f;
		};

		struct Triangle
		{
			int id = -1;
			Vertex* v0 = nullptr;
			Vertex* v1 = nullptr;
			Vertex* v2 = nullptr;
			glm::vec3 centroid = glm::zero<glm::vec3>();
			glm::vec3 normal = glm::zero<glm::vec3>();
		};

		class KDTreeNode
		{
		public:
			KDTreeNode(Vertex* vertex) : v(vertex) {}

			inline Vertex* GetVertex() const { return v; }
			inline KDTreeNode* GetLeft() const { return left ; }
			inline void SetLeft(KDTreeNode* node) { left = node; }
			inline KDTreeNode* GetRight() const { return right; }
			inline void SetRight(KDTreeNode* node) { right = node; }

		private:
			Vertex* v;
			KDTreeNode* left = nullptr;
			KDTreeNode* right = nullptr;

		public:
			friend class KDTree;
		};

		class KDTree
		{
		public:
			KDTree() {}

			void Clear();

			void Insert(Vertex* vertex);

			Vertex* FindNearestNeighbor(const glm::vec3& query);
			KDTreeNode* FindNearestNeighborNode(const glm::vec3& query);

			vector<Vertex*> RangeSearch(const glm::vec3& query, float squaredRadius) const;

			inline bool IsEmpty() const { return nullptr == root; }

		private:
			KDTreeNode* root = nullptr;
			KDTreeNode* nearestNeighborNode = nullptr;
			Vertex* nearestNeighbor = nullptr;
			float nearestNeighborDistance = FLT_MAX;

			void ClearRecursive(KDTreeNode* node);
			KDTreeNode* InsertRecursive(KDTreeNode* node, Vertex* vertex, int depth);
			void FindNearestNeighborRecursive(KDTreeNode* node, const glm::vec3& query, int depth);
			void RangeSearchRecursive(KDTreeNode* node, const glm::vec3& query, float squaredRadius, std::vector<Vertex*>& result, int depth) const;
		};

	public:
		VETM(const string& name, Mesh* mesh);
		~VETM();

		void Clear();
		void Clone(Mesh& clone);

		Vertex* GetVertex(const glm::vec3& position);
		Vertex* AddVertex(const glm::vec3& position, const glm::vec3& normal);

		Edge* GetEdge(Vertex* v0, Vertex* v1);
		Edge* AddEdge(Vertex* v0, Vertex* v1);
		Vertex* GetCommonVertex(Edge* e0, Edge* e1);

		Triangle* GetTriangle(Vertex* v0, Vertex* v1, Vertex* v2);
		Triangle* AddTriangle(Vertex* v0, Vertex* v1, Vertex* v2);
		void RemoveTriangle(Triangle* triangle);

		set<Vertex*> GetAdjacentVertices(Vertex* vertex);

		set<Vertex*> GetVerticesInRadius(const glm::vec3& position, float radius);

		//float GetDistanceFromEdge(Edge* edge, const glm::vec3& position);

		tuple<glm::vec3, glm::vec3, glm::vec3> GetTrianglePoints(Triangle* triangle);
		glm::vec3 GetTriangleCentroid(Triangle* triangle);
		float GetTriangleArea(Triangle* triangle);

		void FlipTriangle(Triangle* triangle);

		glm::vec3 GetNearestPointOnEdge(Edge* edge, const glm::vec3& position);
		Vertex* GetNearestVertexOnTriangle(Triangle* triangle, const glm::vec3& position);
		//Edge* GetNearestEdgeOnTriangle(Triangle* triangle, const glm::vec3& position);
		set<Triangle*> GetAdjacentTrianglesByEdge(Triangle* triangle);
		set<Triangle*> GetAdjacentTrianglesByVertex(Triangle* triangle);
		set<Triangle*> GetConnectedTriangles(Triangle* triangle);

		vector<Mesh*> SeparateConnectedGroup();

		vector<vector<Edge*>> GetBorderEdges();
		void FillTrianglesToMakeBorderSmooth(float degreeMax);
		void ExtrudeBorder(const glm::vec3& direction, int segments);
		void GenerateBase();
		void GenerateBaseWithHollow();
		void DeleteSelfintersectingTriangles();

		inline float GetTotalArea() const { return totalArea; }

		inline const vector<Vertex*>& GetVertices() const { return vertices; }
		inline const set<Edge*>& GetEdges() const { return edges; }
		inline const set<Triangle*>& GetTriangles() const { return triangles; }

	private:
		KDTree kdtree;

		int vid = 0;
		int eid = 0;
		int tid = 0;

		vector<Vertex*> vertices;
		set<Edge*> edges;
		map<tuple<Vertex*, Vertex*>, Edge*> edgeMapping;
		set<Triangle*> triangles;
		map<tuple<Edge*, Edge*, Edge*>, Triangle*> triangleMapping;
		float totalArea = 0.0;

	private:
		Mesh* mesh = nullptr;
	};
}
