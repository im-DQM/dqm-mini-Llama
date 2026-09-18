#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

namespace dqm {

// 压缩前缀树：存一堆 token 序列，回答
// 「这串 token 的开头，在我存过的序列里，最长能匹配到第几个」。
//
// 只存 token 序列本身 —— 不碰 KV、不碰权重、不碰任何数值。
// 它给的是一个数字，怎么用那个数字是调用方的事。
class RadixTree {
 public:
  RadixTree();
  ~RadixTree();

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete; 
  RadixTree(RadixTree&&) noexcept;
  RadixTree& operator=(RadixTree&&) noexcept;

  void Clear();
  bool empty() const;
  size_t size() const;

  void Insert(const std::vector<int>& tokens);
  size_t LongestPrefix(const std::vector<int>& tokens) const;

 private:
  struct Node {
    std::vector<int> key;
    bool terminal = false;
    std::map<int, std::unique_ptr<Node>> children;
  };

  static size_t CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                   const std::vector<int>& b);
  void InsertInto(Node* parent, std::vector<int> suffix);

  Node root_;
  size_t terminal_count_ = 0;
};

}  // namespace dqm
