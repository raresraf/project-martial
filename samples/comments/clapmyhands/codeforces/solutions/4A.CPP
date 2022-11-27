#include <iostream>

int main() {
  int x;
  std::cin >> x;
  // check for even
  if (x & 1 || x <= 2) {
    std::cout << "NO" << std::endl;
  } else {
    std::cout << "YES" << std::endl;
  }
  return 0;
}