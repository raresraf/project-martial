#include <iostream>

int main() {
  int n;

  int c1 = 0;
  int c2 = 0;
  int c3 = 0;

  int temp;

  std::cin >> n;

  for (int i = 0; i < n; i++) {
    std::cin >> temp;
    c1 += temp;

    std::cin >> temp;
    c2 += temp;

    std::cin >> temp;
    c3 += temp;
  }

  std::cout << ((!c1 && !c2 && !c3) ? "YES" : "NO") << std::endl;

  return 0;
}
