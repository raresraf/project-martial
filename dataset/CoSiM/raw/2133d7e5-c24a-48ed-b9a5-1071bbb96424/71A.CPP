#include <iostream>

int main() {
  int n;
  std::cin >> n;
  std::string s;
  while (n--) {
    std::cin >> s;
    int size = s.size();
    if (size > 10) {
      std::cout << s[0] << size - 2 << s[size - 1] << '\n';
    } else {
      std::cout << s << '\n';
    }
  }
}
