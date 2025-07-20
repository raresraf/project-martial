#include <iostream>

int main() {
  int k, n, w, cost = 0;
  std::cin >> k >> n >> w;

  for (int i = 1; i < w + 1; i++) {
    cost += (i * k);
  }

  if (n - cost > 0) {
    std::cout << 0;
  } else {
    std::cout << abs(n - cost);
  }

  return 0;
}