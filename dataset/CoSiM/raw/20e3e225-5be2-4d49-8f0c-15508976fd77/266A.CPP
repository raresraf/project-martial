#include <iostream>

int main() {
  int n;
  std::cin >> n;
  char prev, temp;
  std::cin >> prev;
  int count = 0;
  n--;
  while (n--) {
    std::cin >> temp;
    if (temp != prev) {
      prev = temp;
    } else if (temp == prev) {
      ++count;
    }
  }
  std::cout << count << '\n';
}
