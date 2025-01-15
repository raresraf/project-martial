#include <iostream>

int main() {
  int n;
  int a, b;
  int current = 0;
  int maxNumber = 0;

  std::cin >> n;

  for (int i = 0; i < n; i++) {
    std::cin >> a >> b;
    current += -a + b;

    maxNumber = (current > maxNumber ? current : maxNumber);
  }

  std::cout << maxNumber << std::endl;

  return 0;
}
