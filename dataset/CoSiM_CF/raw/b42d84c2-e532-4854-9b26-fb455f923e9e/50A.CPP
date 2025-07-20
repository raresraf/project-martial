/**
 * Subject: Domino piling
 * URL: https://codeforces.com/problemset/problem/50/A
 */
#include <iostream>

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);
  std::cout.tie(0);

  int m, n;
  std::cin >> m >> n;
  std::cout << (m * n) / 2 << "\n";
  return 0;
}