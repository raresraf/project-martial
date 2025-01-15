#include <iostream>

int main() {
  char c;
  int count = 0;
  std::string s = "hello";
  while (std::cin >> c) {
    if (c == s[count]) {
      ++count;
      if (count == s.size()) {
        std::cout << "YES\n";
        return 0;
      }
    }
  }
  std::cout << "NO\n";
}
