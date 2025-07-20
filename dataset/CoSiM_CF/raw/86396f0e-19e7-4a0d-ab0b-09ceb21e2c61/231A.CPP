#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  int n;
  std::cin >> n;
  int a, b, c, count = 0;
  while (n--) {
    std::cin >> a >> b >> c;
    if ((a + b + c) >= 2) {
      count++;
    }
  }
  std::cout << count << std::endl;
}