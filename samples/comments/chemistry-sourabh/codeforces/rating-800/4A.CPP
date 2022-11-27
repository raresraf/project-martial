//
// Created by Sourabh Bollapragada on 6/3/20.
//

// Problem link https://codeforces.com/problemset/problem/4/A

#include <iostream>

int main() {
  int w;
  std::cin >> w;

  if (w > 2 and w % 2 == 0) {
    std::cout << "YES" << std::endl;
  } else {
    std::cout << "NO" << std::endl;
  }
}
