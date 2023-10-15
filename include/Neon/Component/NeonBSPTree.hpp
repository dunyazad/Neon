#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	template<typename T>
	class BSPTreeNode
	{
	public:
		BSPTreeNode(Mesh* mesh, const T& t, size_t index)
			: mesh(mesh), t(t), index(index)
		{
		}

		~BSPTreeNode()
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
					positive = new BSPTreeNode<T>(mesh, t, index);
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
					negative = new BSPTreeNode<T>(mesh, t, index);
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
		T t = nullptr;
		size_t index = 0;

		BSPTreeNode* positive = nullptr;
		BSPTreeNode* negative = nullptr;
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
		BSPTreeNode<T>* root = nullptr;

		void Build()
		{
			auto vb = mesh->GetVertexBuffer()->GetElements();
			auto nov = vb.size();
			for (size_t i = 0; i < nov; i++)
			{
				auto& v = vb[i];

				if (nullptr == root)
				{
					root = new BSPTreeNode<glm::vec3>(mesh, v, i);
				}
				else
				{
					root->Insert(v, i);
				}
			}
		}

		void Clear()
		{
			if (nullptr != root)
			{
				root->Clear();
				SAFE_DELETE(root);
			}
		}

		void Traverse(BSPTreeNode<T>* node, function<void(BSPTreeNode<T>*)> callback, function<void()> finishCallback)
		{
			stack<BSPTreeNode<T>*> nodes;
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

		BSPTreeNode<T>* GetNearestNode(BSPTreeNode<T>* currentNode, const T& target, BSPTreeNode<T>* nearestNode)
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
