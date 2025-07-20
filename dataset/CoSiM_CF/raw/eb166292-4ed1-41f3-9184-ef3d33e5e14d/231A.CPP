//#include <bits/stdc++>
#include <iostream>

int main() {
  int total = 0;
  int a, b, c;
  a = 0;
  b = 0;
  c = 0;
  int j = 0;
  int total_to_use = 0;
  std::cin >> total;
  for (int i = 0; i < total; i++) {
    std::cin >> a;
    std::cin >> b;
    std::cin >> c;
    j += a + b + c;
    if (j > 1)
      total_to_use++;
    j = 0;
    a = 0;
    b = 0;
    c = 0;
  }
  std::cout << total_to_use;
  return (0);
}
