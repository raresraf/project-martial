//
// Created by Sourabh Bollapragada on 6/3/20.
//

// Problem Link https://codeforces.com/problemset/problem/71/A

#include <iostream>

int main() {
  int n;
  std::string word;
  std::cin >> n;
  for (int i = 0; i < n; i++) {
    std::cin >> word;
    if (word.length() <= 10) {
      std::cout << word << std::endl;
    } else {
      std::cout << word[0] << word.length() - 2 << word[word.length() - 1]
                << std::endl;
    }
  }
}