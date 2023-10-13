#include <Neon/Component/NeonBSPTree.h>
#include <Neon/Component/NeonMesh.h>

namespace Neon
{
	struct BSPVertex
	{
		int index = -1;
		glm::vec3 position = glm::zero<glm::vec3>();
	};

	struct BSPTriangle
	{
		BSPVertex* v0 = nullptr;
		BSPVertex* v1 = nullptr;
		BSPVertex* v2 = nullptr;

		glm::vec3 normal = glm::zero<glm::vec3>();
		glm::vec3 centroid = glm::zero<glm::vec3>();
	};

	class BSPTreeNode
	{
	public:
		BSPTreeNode(Mesh* mesh, BSPTriangle* triangle)
			: triangle(triangle)
		{
		}

		~BSPTreeNode()
		{
		}

		void InsertTriangle(BSPTriangle* triangle)
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

	private:
		Mesh* mesh = nullptr;
		BSPTriangle* triangle = nullptr;
		
		BSPTreeNode* positive = nullptr;
		BSPTreeNode* negative = nullptr;

		VertexBufferObject<float>* vertexBuffer = nullptr;
		VertexBufferObject<GLuint>* indexBuffer = nullptr;
	};


	BSPTree::BSPTree(const string& name, Mesh* mesh)
		: ComponentBase(name)
	{
	}

	BSPTree::~BSPTree()
	{
	}

	void BSPTree::Build()
	{
		
	}

	void BSPTree::Clear()
	{

	}
}
