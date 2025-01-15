#include <iostream>
#include <string>

int main() {
  int n;
  int x = 0;
  std::cin >> n;
  for (int i = 0; i < n; i++) {
    std::string s;
    std::cin >> s;
    if (s.find("++") != std::string::npos) {
      x += 1;
    }
    if (s.find("--") != std::string::npos) {
      x -= 1;
    }
  }
  std::cout << x << std::endl;
}