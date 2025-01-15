#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  int k, n, w;
  std::cin >> k >> n >> w;
  int cost = (w * (w + 1) / 2) * k - n;
  cost = cost < 0 ? 0 : cost;
  std::cout << cost << std::endl;
  return 0;
}