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

    explicit DataNode(ColumnNode* column)
      : left(this)
      , right(this)
      , up(this)
      , down(this)
      , column(column)
    {}

    virtual ~DataNode() = default;

    void cover();

    void uncover();

    void insertBelow(DataNode* node)
    {
        node->down = this->down;
        node->down->up = node;
        node->up = this;
        this->down = node;
    }

    void insertRight(DataNode* node)
    {
        node->right = this->right;
        node->right->left = node;
        node->left = this;
        this->right = node;
    }
};

struct ColumnNode : public DataNode
{
    size_t size;
    size_t index;

    explicit ColumnNode(size_t index)
      : DataNode(this)
      , size(0)
      , index(index)
    {}
};

void
DataNode::cover()
{
    this->left->right = this->right;
    this->right->left = this->left;

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

struct DlxSolver::Impl
{
    std::vector<std::unique_ptr<DataNode>> nodes;
    std::vector<DataNode*> solution;
    DataNode* head = nullptr;

    std::unique_ptr<std::vector<std::vector<size_t>>> solve(const std::vector<std::vector<int>>& constraintMatrix)
    {

        nodes.push_back(std::make_unique<ColumnNode>(0));

        encodeConstraints(constraintMatrix);
        auto result = search(0);

        nodes.clear();
        solution.clear();
        head = nullptr;

        return result;
    }

    void solveAll(const std::vector<std::vector<int>>& constraintMatrix,
                  const std::function<void(std::unique_ptr<std::vector<std::vector<size_t>>>)>& resultCollector)
    {

        nodes.push_back(std::make_unique<ColumnNode>(0));

        encodeConstraints(constraintMatrix);
        searchAll(0, resultCollector);

        nodes.clear();
        solution.clear();
        head = nullptr;
    }

    DataNode* createDataNode(ColumnNode* column)
    {
        nodes.push_back(std::make_unique<DataNode>(column));
        return nodes.back().get();
    }

    ColumnNode* createColNode(size_t index)
    {
        nodes.push_back(std::make_unique<ColumnNode>(index));
        return dynamic_cast<ColumnNode*>(nodes.back().get());
    }

    void encodeConstraints(const std::vector<std::vector<int>>& constraintMatrix)
    {
        head = createDataNode(nullptr);

        // assumes that all rows in the matrix are of the same size.
        size_t num_cols = constraintMatrix[0].size();

        // Create column nodes
        std::vector<ColumnNode*> columns;
        columns.reserve(num_cols);
        for (size_t i = 0; i < num_cols; i++) {
            auto col = createColNode(i);
            // head->left is the last/rightmost node in the list because the lists are circular
            head->left->insertRight(col);
            columns.push_back(col);
        }

        for (auto row : constraintMatrix) {
            DataNode* prev = nullptr;
            for (size_t i = 0; i < num_cols; i++) {
                int value = row[i];

                // Only cells with 1 are represented as nodes
                if (value == 1) {
                    auto col = columns[i];
                    auto n = createDataNode(col);
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
    }

    std::unique_ptr<std::vector<std::vector<size_t>>> decodeSolution()
    {
        auto resultRows = std::make_unique<std::vector<std::vector<size_t>>>();
        resultRows->reserve(solution.size());
        for (auto node : solution) {
            std::vector<size_t> ones;

            auto i = node;
            do {
                ones.push_back(i->column->index);
                i = i->right;
            } while (i != node);

            std::sort(std::begin(ones), std::end(ones));

            resultRows->push_back(ones);
        }

        return resultRows;
    }

    [[nodiscard]] ColumnNode* selectColumn() const
    {
#ifdef USE_HEURISTIC
        return dynamic_cast<ColumnNode*>(head->right);
#else
        auto* best = dynamic_cast<ColumnNode*>(head->right);
        for (auto j = head->right; head != j; j = j->right) {
            auto k = dynamic_cast<ColumnNode*>(j);
            if (k->size < best->size) {
                best = k;
            }
        }
        return best;
#endif
    }

    std::unique_ptr<std::vector<std::vector<size_t>>> search(int k)
    {

        if (head == head->right) {
            return decodeSolution();
        }

        auto c = selectColumn();
        c->cover();

        for (auto r = c->down; r != c; r = r->down) {
            solution.push_back(r);

            for (auto j = r->right; j != r; j = j->right) {
                j->column->cover();
            }

            auto result = search(k + 1);
            if (result != nullptr) {
                return result;
            }

            r = solution.back();
            solution.pop_back();
            c = r->column;

            for (auto j = r->left; j != r; j = j->left) {
                j->column->uncover();
            }
        }
        c->uncover();

        return nullptr;
    }

    void searchAll(int k, const std::function<void(std::unique_ptr<std::vector<std::vector<size_t>>>)>& resultCollector)
    {
        if (head == head->right) {
            resultCollector(decodeSolution());
        }

        auto c = selectColumn();
        c->cover();

        for (auto r = c->down; r != c; r = r->down) {
            solution.push_back(r);

            for (auto j = r->right; j != r; j = j->right) {
                j->column->cover();
            }
            searchAll(k + 1, resultCollector);

            r = solution.back();
            solution.pop_back();
            c = r->column;

            for (auto j = r->left; j != r; j = j->left) {
                j->column->uncover();
            }
        }
        c->uncover();
    }
};
DlxSolver::DlxSolver()
  : impl(std::make_unique<Impl>())
{}
DlxSolver::~DlxSolver() = default;
DlxSolver::DlxSolver(DlxSolver&&) noexcept = default;
DlxSolver&
DlxSolver::operator=(DlxSolver&&) noexcept = default;
std::unique_ptr<std::vector<std::vector<size_t>>>
DlxSolver::solve(const std::vector<std::vector<int>>& constraintMatrix)
{
    return impl->solve(constraintMatrix);
}
void
DlxSolver::solve(const std::vector<std::vector<int>>& constraintMatrix,
                 const std::function<void(std::unique_ptr<std::vector<std::vector<size_t>>>)>& resultCollector)
{
    impl->solveAll(constraintMatrix, resultCollector);
}
