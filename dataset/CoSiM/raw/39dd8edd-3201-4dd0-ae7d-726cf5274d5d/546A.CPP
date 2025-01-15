#include <bits/stdc++.h>

int main() {
  int k, n, w;
  std::cin >> k >> n >> w;

  // Total cost of all bananas
  int cost = 0;
  for (int i = 1; i <= w; i++) {
    cost += (i * k);
  }
  if (cost - n <= 0) {
    std::cout << 0;
    return 0;
  }
  std::cout << cost - n;

  return 0;
}