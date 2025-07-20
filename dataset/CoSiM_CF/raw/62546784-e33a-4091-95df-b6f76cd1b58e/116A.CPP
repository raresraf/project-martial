#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  int n;
  std::cin >> n;

  int out, in, count = 0;
  int max = 0;
  while (n--) {
    std::cin >> out >> in;
    count = count - out + in;
    if (count > max)
      max = count;
  }
  std::cout << max << std::endl;
  return 0;
}