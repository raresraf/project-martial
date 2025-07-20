#include <iostream>

int main() {
  int n, d, m;
  std::cin >> n;
  int people = 0;
  int capacity = 0;
  for (int i = 0; i < n; i++) {
    std::cin >> d >> m;
    people += m - d;
    capacity = capacity < people and i != n - 1 ? people : capacity;
  }
  std::cout << capacity << std::endl;
}
