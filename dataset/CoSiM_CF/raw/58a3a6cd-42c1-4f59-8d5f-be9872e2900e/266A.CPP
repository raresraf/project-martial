#include <iostream>

int main() {
  unsigned int n, nb = 0;
  std::cin >> n;

  char c, pc;
  for (int i = 0; i < n; i++) {
    std::cin >> c;
    if (i > 0 and c == pc)
      nb++;
    pc = c;
  }
  std::cout << nb << "\n";
}
