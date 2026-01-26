#include <iostream>

int main() {
  char c;
  bool done = false;
  while (std::cin >> c) {
    if (!done) {
      if ((c >= 'a') && (c <= 'z'))
        c += 'A' - 'a';
      done = true;
    }
    std::cout << c;
  }
  std::cout << '\n';
}
