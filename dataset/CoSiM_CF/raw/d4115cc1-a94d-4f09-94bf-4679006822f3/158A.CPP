//
// Created by Sourabh Bollapragada on 6/3/20.
//

// Problem Link https://codeforces.com/problemset/problem/158/A

#include <iostream>

int main() {
  int n, k;
  std::cin >> n >> k;
  int s[n];

  for (int i = 0; i < n; i++) {
    std::cin >> s[i];
  }

  int c = 0;
  for (int i = 0; i < n; i++) {
    if (c < k && s[i] > 0) {
      c++;
    } else if (s[i] > 0 && s[i] == s[k - 1]) {
      c++;
    }
  }

  std::cout << c << std::endl;
}