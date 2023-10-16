#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	struct BSPTriangle
	{
		glm::vec3 v0;
		glm::vec3 v1;
		glm::vec3 v2;

		GLuint i0;
		GLuint i1;
		GLuint i2;

		glm::vec3 centroid;
		glm::vec3 normal;
	};

	template<typename T>
	class BSPTreeVertexNode
	{
	public:
		BSPTreeVertexNode(Mesh* mesh, const T& t, size_t index)
			: mesh(mesh), t(t), index(index)
		{
		}

		~BSPTreeVertexNode()
		{
		}

		void Insert(const T& t, size_t index)
		{
			if (t < this->t)
			{
				if (nullptr != positive)
				{
					positive->Insert(t, index);
				}
				else
				{
					positive = new BSPTreeVertexNode<T>(mesh, t, index);
				}
			}
			else if (t > this->t)
			{
				if (nullptr != negative)
				{
					negative->Insert(t, index);
				}
				else
				{
					negative = new BSPTreeVertexNode<T>(mesh, t, index);
				}
			}
		}

		/*	void InsertTriangle(BSPTriangle* triangle)
			{
				float sd0 = glm::dot(triangle->v0->position - this->triangle->centroid, this->triangle->normal);
				float sd1 = glm::dot(triangle->v1->position - this->triangle->centroid, this->triangle->normal);
				float sd2 = glm::dot(triangle->v2->position - this->triangle->centroid, this->triangle->normal);

				if (sd0 > 0 && sd1 > 0 && sd2 > 0)
				{
					if (nullptr != positive)
					{
						positive->InsertTriangle(triangle);
					}
					else
					{
						positive = new BSPTreeNode(mesh, triangle);
					}
				}
				else if (sd0 < 0 && sd1 < 0 && sd2 < 0)
				{
					if (nullptr != negative)
					{
						negative->InsertTriangle(triangle);
					}
					else
					{
						negative = new BSPTreeNode(mesh, triangle);
					}
				}
				else if (abs(sd0) <= FLT_EPSILON && abs(sd1) <= FLT_EPSILON && abs(sd2) <= FLT_EPSILON)
				{
					cout << "coplanar" << endl;
				}
				else
				{
					cout << "intersection" << endl;
				}
			}*/

		void Clear()
		{
			if (nullptr != positive)
			{
				positive->Clear();
				SAFE_DELETE(positive);
			}
			if (nullptr != negative)
			{
				negative->Clear();
				SAFE_DELETE(negative);
			}
		}

	//private:
		Mesh* mesh = nullptr;
		T t;
		size_t index = 0;

		BSPTreeVertexNode<T>* positive = nullptr;
		BSPTreeVertexNode<T>* negative = nullptr;
	};

	template<typename T>
	class BSPTreeTriangleNode
	{
	public:
		BSPTreeTriangleNode(Mesh* mesh, const T& t, size_t index)
			: mesh(mesh), t(t), index(index)
		{
		}

		~BSPTreeTriangleNode()
		{
		}

		//void Insert(const T& t, size_t index)
		//{
		//	if (t < this->t)
		//	{
		//		if (nullptr != positive)
		//		{
		//			positive->Insert(t, index);
		//		}
		//		else
		//		{
		//			positive = new BSPTreeVertexNode<T>(mesh, t, index);
		//		}
		//	}
		//	else if (t > this->t)
		//	{
		//		if (nullptr != negative)
		//		{
		//			negative->Insert(t, index);
		//		}
		//		else
		//		{
		//			negative = new BSPTreeVertexNode<T>(mesh, t, index);
		//		}
		//	}
		//}

		void Insert(const T& t, size_t index)
		{
			float sd0 = glm::dot(t.v0 - this->t.centroid, this->t.normal);
			float sd1 = glm::dot(t.v1 - this->t.centroid, this->t.normal);
			float sd2 = glm::dot(t.v2 - this->t.centroid, this->t.normal);

			if (sd0 > 0 && sd1 > 0 && sd2 > 0)
			{
				if (nullptr != positive)
				{
					positive->Insert(t, index);
				}
				else
				{
					positive = new BSPTreeTriangleNode<T>(mesh, t, index);
				}
			}
			else if (sd0 < 0 && sd1 < 0 && sd2 < 0)
			{
				if (nullptr != negative)
				{
					negative->Insert(t, index);
				}
				else
				{
					negative = new BSPTreeTriangleNode<T>(mesh, t, index);
				}
			}
			else if (abs(sd0) <= FLT_EPSILON && abs(sd1) <= FLT_EPSILON && abs(sd2) <= FLT_EPSILON)
			{
				cout << "coplanar" << endl;
			}
			else
			{
				//cout << "intersection" << endl;

				glm::vec3 i01, i12, i20;
				auto b01 = Intersection::LinePlaneIntersection(t.v0, t.v1, this->t.centroid, this->t.normal, i01);
				auto b12 = Intersection::LinePlaneIntersection(t.v1, t.v2, this->t.centroid, this->t.normal, i12);
				auto b20 = Intersection::LinePlaneIntersection(t.v2, t.v0, this->t.centroid, this->t.normal, i20);

				if (b01 && b12 && b20)
				{
					cout << "!!!!!!!!!! coplanar" << endl;
				}
				else if (b01 && b12)
				{
					auto ii01 = (GLuint)mesh->AddVertex(i01);
					auto ii12 = (GLuint)mesh->AddVertex(i12);

					if ((i01 == t.v0 || i01 == t.v1 || i01 == t.v2) &&
						(i12 == t.v0 || i12 == t.v1 || i12 == t.v2))
					{
						return;
						//return result;
					}

					if (i01 == t.v0)
					{
						mesh->AddTriangle(t.i0, t.i1, ii12);
						mesh->AddTriangle(t.i2, t.i0, ii12);

						//RemoveTriangle(triangle);
					}
					else if (i01 == t.v1)
					{
						//return result;
						return;
					}
					else if (i12 == t.v1)
					{
						//return result;
						return;
					}
					else if (i12 == t.v2)
					{
						mesh->AddTriangle(t.i2, t.i0, ii12);
						mesh->AddTriangle(t.i1, t.i2, ii12);

						//RemoveTriangle(triangle);
					}
					else
					{
						mesh->AddTriangle(t.i0, ii01, ii12);
						mesh->AddTriangle(t.i0, ii12, t.i2);
						mesh->AddTriangle(t.i1, ii12, ii01);

						//RemoveTriangle(triangle);
					}
				}
				else if (b12 && b20)
				{
					auto ii12 = (GLuint)mesh->AddVertex(i12);
					auto ii20 = (GLuint)mesh->AddVertex(i20);

					if ((i12 == t.v0 || i12 == t.v1 || i12 == t.v2) &&
						(i20 == t.v0 || i20 == t.v1 || i20 == t.v2))
					{
						//return result;
						return;
					}

					if (i12 == t.v1)
					{
						mesh->AddTriangle(t.i0, t.i1, ii20);
						mesh->AddTriangle(t.i1, t.i2, ii20);

						//RemoveTriangle(triangle);
					}
					else if (i12 == t.v2)
					{
						//return result;
						return;
					}
					else if (i20 == t.v2)
					{
						//return result;
						return;
					}
					else if (i20 == t.v0)
					{
						mesh->AddTriangle(t.i0, t.i1, ii12);
						mesh->AddTriangle(t.i2, t.i0, ii12);

						//RemoveTriangle(triangle);
					}
					else
					{
						mesh->AddTriangle(t.i0, t.i1, ii12);
						mesh->AddTriangle(t.i0, ii12, ii20);
						mesh->AddTriangle(t.i2, ii20, ii12);

						//RemoveTriangle(triangle);
					}
				}
				else if (b20 && b01)
				{
					auto ii20 = (GLuint)mesh->AddVertex(i20);
					auto ii01 = (GLuint)mesh->AddVertex(i01);

					if ((i20 == t.v0 || i20 == t.v1 || i20 == t.v2) &&
						(i01 == t.v0 || i01 == t.v1 || i01 == t.v2))
					{
						//return result;
						return;
					}

					if (i20 == t.v2)
					{
						mesh->AddTriangle(t.i2, t.i0, ii01);
						mesh->AddTriangle(t.i1, t.i2, ii01);

						//RemoveTriangle(triangle);
					}
					else if (i20 == t.v0)
					{
						//return result;
						return;
					}
					else if (i01 == t.v0)
					{
						//return result;
						return;
					}
					else if (i01 == t.v1)
					{
						mesh->AddTriangle(t.i0, t.i1, ii20);
						mesh->AddTriangle(t.i1, t.i2, ii20);

						//RemoveTriangle(triangle);
					}
					else
					{
						mesh->AddTriangle(t.i1, t.i2, ii20);
						mesh->AddTriangle(t.i1, ii20, ii01);
						mesh->AddTriangle(t.i0, ii01, ii20);

						//RemoveTriangle(triangle);
					}
				}
				else
				{
					cout << "???" << endl;
				}
			}
		}

		void Clear()
		{
			if (nullptr != positive)
			{
				positive->Clear();
				SAFE_DELETE(positive);
			}
			if (nullptr != negative)
			{
				negative->Clear();
				SAFE_DELETE(negative);
			}
		}

		//private:
		Mesh* mesh = nullptr;
		T t;
		size_t index = 0;

		BSPTreeTriangleNode<T>* positive = nullptr;
		BSPTreeTriangleNode<T>* negative = nullptr;
	};

	template<typename T>
	class BSPTree : public ComponentBase
	{
	public:
		BSPTree(const string& name, Mesh* mesh)
			: ComponentBase(name), mesh(mesh)
		{
		}

		virtual ~BSPTree()
		{
		}

		Mesh* mesh = nullptr;
		BSPTreeVertexNode<T>* vertexRoot = nullptr;
		BSPTreeTriangleNode<T>* triangleRoot = nullptr;

		void BuildVertices()
		{
			auto vb = mesh->GetVertexBuffer()->GetElements();
			auto nov = vb.size();
			for (size_t i = 0; i < nov; i++)
			{
				auto& v = vb[i];

				if (nullptr == vertexRoot)
				{
					vertexRoot = new BSPTreeVertexNode<glm::vec3>(mesh, v, i);
				}
				else
				{
					vertexRoot->Insert(v, i);
				}
			}
		}

		void BuildTriangles()
		{
			auto noi = mesh->GetIndexBuffer()->Size();
			for (size_t i = 0; i < noi / 3; i++)
			{
				GLuint i0, i1, i2;
				mesh->GetIndex(i * 3 + 0, i0);
				mesh->GetIndex(i * 3 + 1, i1);
				mesh->GetIndex(i * 3 + 2, i2);

				auto v0 = mesh->GetVertex(i0);
				auto v1 = mesh->GetVertex(i1);
				auto v2 = mesh->GetVertex(i2);

				glm::vec3 centroid = (v0 + v1 + v2) / 3.0f;
				auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));


				if (nullptr == triangleRoot)
				{
					triangleRoot = new BSPTreeTriangleNode<T>(mesh, {v0, v1, v2, i0, i1, i2, centroid, normal}, i / 3);
				}
				else
				{
					triangleRoot->Insert({ v0, v1, v2, i0, i1, i2, centroid, normal }, i / 3);
				}
			}
		}

		void Clear()
		{
			if (nullptr != vertexRoot)
			{
				vertexRoot->Clear();
				SAFE_DELETE(vertexRoot);
			}

			if (nullptr != triangleRoot)
			{
				triangleRoot->Clear();
				SAFE_DELETE(triangleRoot);
			}
		}

		void Traverse(BSPTreeVertexNode<T>* node, function<void(BSPTreeVertexNode<T>*)> callback, function<void()> finishCallback)
		{
			stack<BSPTreeVertexNode<T>*> nodes;
			nodes.push(node);
			while (nodes.empty() == false)
			{
				auto currentNode = nodes.top();
				nodes.pop();

				callback(currentNode);

				if (currentNode->positive)
				{
					nodes.push(currentNode->positive);
				}

				if (currentNode->negative)
				{
					nodes.push(currentNode->negative);
				}
			}

			finishCallback();
		}

		void Traverse(BSPTreeTriangleNode<T>* node, function<void(BSPTreeTriangleNode<T>*)> callback, function<void()> finishCallback)
		{
			stack<BSPTreeTriangleNode<T>*> nodes;
			nodes.push(node);
			while (nodes.empty() == false)
			{
				auto currentNode = nodes.top();
				nodes.pop();

				callback(currentNode);

				if (currentNode->positive)
				{
					nodes.push(currentNode->positive);
				}

				if (currentNode->negative)
				{
					nodes.push(currentNode->negative);
				}
			}

			finishCallback();
		}

		BSPTreeVertexNode<T>* GetNearestNode(BSPTreeVertexNode<T>* currentNode, const T& target, BSPTreeVertexNode<T>* nearestNode)
		{
			if (currentNode == nullptr)
			{
				return nearestNode;
			}

			float currentDistance = glm::distance(currentNode->t, target);
			float nearestDistance = glm::distance(nearestNode->t, target);

			if (currentDistance < nearestDistance)
			{
				nearestNode = currentNode;
			}

			if (currentNode->t < target)
			{
				nearestNode = GetNearestNode(currentNode->positive, target, nearestNode);
				nearestNode = GetNearestNode(currentNode->negative, target, nearestNode);
			}
			else
			{
				nearestNode = GetNearestNode(currentNode->negative, target, nearestNode);
				nearestNode = GetNearestNode(currentNode->positive, target, nearestNode);
			}

			return nearestNode;
		}
	};
}
