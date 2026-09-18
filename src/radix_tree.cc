#include "dqm_llama/radix_tree.h"

#include <cstddef>
#include <memory>
#include <utility>

namespace dqm {

RadixTree::RadixTree() = default;
RadixTree::~RadixTree() = default;
RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;

void RadixTree::Clear() {
  root_.children.clear();
  root_.terminal = false;
  root_.key.clear();
  terminal_count_ = 0;
}

bool RadixTree::empty() const {
  return terminal_count_ == 0;
}

size_t RadixTree::size() const {
  return terminal_count_;
}

size_t RadixTree::CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                     const std::vector<int>& b) {
  size_t i = 0;
  size_t as = a.size();
  size_t bs = b.size();
  while (i + a_offset < as && i < bs) {
    if (a[i + a_offset] == b[i]) {
      i ++;
    } else {
      return i;
    }
  }
  return i;
}

void RadixTree::Insert(const std::vector<int>& tokens) {
  if (tokens.empty()) {
    if (!root_.terminal) {
      root_.terminal = true;
      terminal_count_ ++;
    }
    return;
  }
  InsertInto(&root_, tokens);
}

void RadixTree::InsertInto(Node* parent, std::vector<int> suffix) {
  // ① 这个 token 还没有对应的子边 → 整段 suffix 就是新边，直接挂一个叶子
  auto child_it = parent->children.find(suffix[0]);
  if (child_it == parent->children.end()) {
    auto child = std::make_unique<Node>();
    child->key = std::move(suffix);
    child->terminal = true;
    // 键取下标 0，是在用代码钉住「键 == key[0]」这个不变量
    parent->children[child->key[0]] = std::move(child);
    terminal_count_ ++;
    return;
  }

  // 必须先把槽位里的 unique_ptr 绑成引用：后面 std::move 走之后就用不了了
  std::unique_ptr<Node>& child = child_it->second;
  const size_t common = CommonPrefixLength(suffix, 0, child->key);

  // ② 整条边都被匹配上
  if (common == child->key.size()) {
    // ②a suffix 也走完了 → 这条序列早就插过，只需补 terminal（幂等）
    if (common == suffix.size()) {
      if (!child->terminal) {
        child->terminal = true;
        terminal_count_ ++;
      }
      return;
    }
    // ②b 还有剩余 → 下探一层继续插
    std::vector<int> remaining(
        suffix.begin() + static_cast<std::ptrdiff_t>(common), suffix.end());
    InsertInto(child.get(), std::move(remaining));
    return;
  }

  // ③ 边在中间就岔开了 → 把这条边劈成两段，让分叉浮出来
  auto split = std::make_unique<Node>();
  split->key.assign(child->key.begin(),
                    child->key.begin() + static_cast<std::ptrdiff_t>(common));

  // 先砍边：砍完 key[0] 才是真正的分叉点。common >= 1 由「find 命中」保证
  child->key.erase(child->key.begin(),
                   child->key.begin() + static_cast<std::ptrdiff_t>(common));
  const int old_child_first = child->key[0];
  split->children[old_child_first] = std::move(child);

  if (common == suffix.size()) {
    // suffix 正好停在劈开处 → split 自己就是终点
    split->terminal = true;
    terminal_count_ ++;
  } else {
    auto new_child = std::make_unique<Node>();
    new_child->key.assign(suffix.begin() + static_cast<std::ptrdiff_t>(common),
                          suffix.end());
    new_child->terminal = true;
    split->children[new_child->key[0]] = std::move(new_child);
    terminal_count_ ++;
  }

  // 放最后：child 原来就住在这个槽里，得先等它被搬进 split->children，
  // 才能让槽位改指向 split（提前覆盖会把它销毁掉）
  child_it->second = std::move(split);
}

size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const {
  size_t pos = 0;
  const Node* node = &root_;
  while (pos < tokens.size()) {
    auto it = node->children.find(tokens[pos]);
    if (it == node->children.end()) {
      return pos;
    }

    const Node* child = it->second.get();
    size_t common = CommonPrefixLength(tokens, pos, child->key);
    pos += common;
    if (common < child->key.size()) {
      return pos;
    } else {
      node = child;
    }
  }
  
  return pos;
}

}  // namespace dqm
