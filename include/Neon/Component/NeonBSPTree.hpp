#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	template<class T>
	class BSPTreeNode
	{
	public:
		BSPTreeNode(Mesh* mesh, T* triangle)
			: triangle(triangle)
		{
		}

		~BSPTreeNode()
		{
		}

		void Insert(T* t)
		{
			if (t->position < this->t->position)
			{
				if (nullptr != positive)
				{
					positive->Insert(t);
				}
				else
				{
					positive = new BSPTreeNode<T>(mesh, t);
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

	private:
		Mesh* mesh = nullptr;
		T* t = nullptr;

		BSPTreeNode* positive = nullptr;
		BSPTreeNode* negative = nullptr;
	};

	template<typename T>
	class BSPTree : public ComponentBase
	{
	public:
		BSPTree(const string& name, Mesh* mesh)
			: ComponentBase(name)
		{
		}

		virtual ~BSPTree()
		{
		}

		Mesh* mesh = nullptr;
		BSPTreeNode<T>* root = nullptr;

		void Build()
		{
		}

		void Clear()
		{
		}
	};
}
