#include <iostream>
#include <string>

int main() {
  int num, l;
  std::string s = "";

  std::cin >> num;
  for (int i = 0; i < num; i++) {
    std::cin >> s;
    l = s.length();
    if (l > 10) {
      std::cout << s[0] << (l - 2) << s[l - 1] << std::endl;
    } else {
      std::cout << s << std::endl;
    }
  }
}