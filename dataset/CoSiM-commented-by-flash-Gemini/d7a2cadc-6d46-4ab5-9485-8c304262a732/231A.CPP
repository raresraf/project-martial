//
// Created by Sourabh Bollapragada on 6/3/20.
//

// Problem Link https://codeforces.com/problemset/problem/231/A

#include <iostream>

int main() {
  int n;
  int p, v, t;
  std::cin >> n;
  int c = 0;
  for (int i = 0; i < n; i++) {
    std::cin >> p >> v >> t;
    if (p + v + t > 1) {
      c++;
    }
  }
  std::cout << c << std::endl;
}
