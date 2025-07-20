#include <bits/stdc++.h>
int main() {
  int k, n, w, cost = 0;
  std::cin >> k >> n >> w;
  for (int i = 1; i <= w; i++)
    cost += k * i;
  (cost - n > 0) ? std::cout << cost - n : std::cout << 0;
}
