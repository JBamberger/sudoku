//
// Created by jannik on 08.08.2021.
//

#include "dlx.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include <array>
#include <iostream>

struct ColumnNode;

struct DataNode
{
    DataNode* left;
    DataNode* right;
    DataNode* up;
    DataNode* down;
    ColumnNode* column;

    explicit DataNode();
    explicit DataNode(ColumnNode* column);
    virtual ~DataNode() = default;

    void cover();
    void uncover();

    void insertBelow(DataNode* node);
    void insertRight(DataNode* node);
};

struct ColumnNode : public DataNode
{
    size_t size;
    size_t index;

    explicit ColumnNode()
      : DataNode(this)
      , size(0)
      , index(0)
    {}
};

class DlxSolver::Impl
{
    std::unique_ptr<DataNode> headPtr = nullptr;
    DataNode* head = nullptr;
    std::vector<ColumnNode> colNodes;
    std::vector<DataNode> dataNodes;
    std::vector<DataNode*> solution;
    size_t numCols = 0;

    void setup(const DlxConstraintMatrix& constraintMatrix);
    void teardown();
    std::unique_ptr<DlxConstraintMatrix> decodeSolution();
    [[nodiscard]] ColumnNode* selectColumn() const;
    std::unique_ptr<DlxConstraintMatrix> search(int k);
    void searchAll(int k, const std::function<void(std::unique_ptr<DlxConstraintMatrix>)>& resultCollector);

  public:
    std::unique_ptr<DlxConstraintMatrix> solve(const DlxConstraintMatrix& constraintMatrix);
    void solveAll(const DlxConstraintMatrix& constraintMatrix,
                  const std::function<void(std::unique_ptr<DlxConstraintMatrix>)>& resultCollector);
};

//*****************************************************************************
// DataNode
//*****************************************************************************

DataNode::DataNode()
  : DataNode(nullptr)
{}

DataNode::DataNode(ColumnNode* column)
  : left(this)
  , right(this)
  , up(this)
  , down(this)
  , column(column)
{}

void
DataNode::cover()
{
    // Unhook this node laterally
    this->left->right = this->right;
    this->right->left = this->left;

    // Unhook all nodes that are connected laterally to nodes in this column.
    // This is equivalent to removing all rows from the matrix where a 1 is
    // placed in the column of this node.
    for (auto i = this->down; i != this; i = i->down) {
        for (auto j = i->right; j != i; j = j->right) {
            j->up->down = j->down;
            j->down->up = j->up;
            j->column->size -= 1;
        }
    }
}

void
DataNode::uncover()
{
    // Re-adds the nodes disconnected in DataNode::cover()
    for (auto i = this->up; i != this; i = i->up) {
        for (auto j = i->left; j != i; j = j->left) {
            j->column->size += 1;
            j->down->up = j;
            j->up->down = j;
        }
    }

    this->right->left = this;
    this->left->right = this;
}

void
DataNode::insertBelow(DataNode* node)
{
    node->down = this->down;
    node->down->up = node;
    node->up = this;
    this->down = node;
}

void
DataNode::insertRight(DataNode* node)
{
    node->right = this->right;
    node->right->left = node;
    node->left = this;
    this->right = node;
}

//*****************************************************************************
// DlxSolver::Impl
//*****************************************************************************

std::unique_ptr<DlxConstraintMatrix>
DlxSolver::Impl::solve(const DlxConstraintMatrix& constraintMatrix)
{
    setup(constraintMatrix);
    auto result = search(0);
    teardown();

    return result;
}

void
DlxSolver::Impl::solveAll(const DlxConstraintMatrix& constraintMatrix,
                          const std::function<void(std::unique_ptr<DlxConstraintMatrix>)>& resultCollector)
{
    setup(constraintMatrix);
    searchAll(0, resultCollector);
    teardown();
}

void
DlxSolver::Impl::setup(const DlxConstraintMatrix& constraintMatrix)
{
    numCols = constraintMatrix.numCols;

    headPtr = std::make_unique<DataNode>();
    head = headPtr.get();

    colNodes = std::vector<ColumnNode>(numCols);

    // Create column nodes
    for (size_t i = 0; i < numCols; i++) {
        ColumnNode* col = &colNodes.at(i);
        col->index = i;

        // head->left is the last/rightmost node in the list because the lists are circular
        head->left->insertRight(col);
    }

    size_t numConstraints = 0;
    for (const auto& row : constraintMatrix.constraints) {
        numConstraints += row.size();
    }
    dataNodes = std::vector<DataNode>(numConstraints);
    size_t nextNode = 0;
    for (const auto& row : constraintMatrix.constraints) {
        DataNode* prev = nullptr;
        for (size_t colIndex : row) {
            auto col = &colNodes[colIndex];
            auto n = &dataNodes[nextNode++];
            n->column = col;

            // col->up is the last node in vertical direction for this column
            col->up->insertBelow(n);
            col->size += 1;

            if (prev == nullptr) {
                prev = n;
            } else {
                prev->insertRight(n);
            }
        }
    }
}

void
DlxSolver::Impl::teardown()
{
    dataNodes.clear();
    colNodes.clear();
    solution.clear();
    headPtr = nullptr;
    head = nullptr;
    numCols = 0;
}

std::unique_ptr<DlxConstraintMatrix>
DlxSolver::Impl::decodeSolution()
{
    auto result = std::make_unique<DlxConstraintMatrix>(0, numCols);
    result->constraints.reserve(solution.size());
    for (auto node : solution) {
        std::vector<size_t> ones;

        auto i = node;
        do {
            ones.push_back(i->column->index);
            i = i->right;
        } while (i != node);

        std::sort(std::begin(ones), std::end(ones));

        result->constraints.push_back(ones);
    }

    return result;
}

ColumnNode*
DlxSolver::Impl::selectColumn() const
{
    auto* best = dynamic_cast<ColumnNode*>(head->right);
    for (auto j = head->right; head != j; j = j->right) {
        auto k = dynamic_cast<ColumnNode*>(j);
        if (k->size < best->size) {
            best = k;
        }
    }
    return best;
}

std::unique_ptr<DlxConstraintMatrix>
DlxSolver::Impl::search(int k)
{
    if (head == head->right) {
        // We've managed to reduce all columns, therefore satisfying all constraints with our selection.
        return decodeSolution();
    }

    // There are columns left. Select one and remove it from the matrix.
    auto c = selectColumn();
    c->cover();

    // Test each row affected by the selected column as part of the solution.
    for (auto r = c->down; r != c; r = r->down) {
        // Use r as part of the current solution attempt
        solution.push_back(r);
        // Remove all rows conflicting with the selected row.
        for (auto j = r->right; j != r; j = j->right) {
            j->column->cover();
        }

        // Recursively, search for a solution in the reduced problem.
        auto result = search(k + 1);
        if (result != nullptr) {
            return result;
        }

        // No solution was found in the recursive call chain. Restore the original state:
        // Re-add the conflicting rows
        for (auto j = r->left; j != r; j = j->left) {
            j->column->uncover();
        }
        // Remove the selected row from the solution
        solution.pop_back();
    }
    // We failed to satisfy the constraints with the current configuration,
    // restore the previous state by re-adding the column and related rows.
    c->uncover();

    return nullptr;
}

void
DlxSolver::Impl::searchAll(int k, const std::function<void(std::unique_ptr<DlxConstraintMatrix>)>& resultCollector)
{
    if (head == head->right) {
        resultCollector(decodeSolution());
        // FIXME: Here should be a 'return', I believe. There are no columns
        //  to select anyway so the loop would terminate immediately anyway as
        //  its up and down links point to itself.
        //  return;
    }

    auto c = selectColumn();
    c->cover();

    for (auto r = c->down; r != c; r = r->down) {
        solution.push_back(r);

        for (auto j = r->right; j != r; j = j->right) {
            j->column->cover();
        }

        searchAll(k + 1, resultCollector);

        for (auto j = r->left; j != r; j = j->left) {
            j->column->uncover();
        }
        solution.pop_back();
    }
    c->uncover();
}

//*****************************************************************************
// DlxSolver
//*****************************************************************************

DlxSolver::DlxSolver()
  : impl(std::make_unique<Impl>())
{}

DlxSolver::~DlxSolver() = default;

DlxSolver::DlxSolver(DlxSolver&&) noexcept = default;

DlxSolver&
DlxSolver::operator=(DlxSolver&&) noexcept = default;

std::unique_ptr<DlxConstraintMatrix>
DlxSolver::solve(const DlxConstraintMatrix& constraintMatrix)
{
    return impl->solve(constraintMatrix);
}

void
DlxSolver::solve(const DlxConstraintMatrix& constraintMatrix,
                 const std::function<void(std::unique_ptr<DlxConstraintMatrix>)>& resultCollector)
{
    impl->solveAll(constraintMatrix, resultCollector);
}
