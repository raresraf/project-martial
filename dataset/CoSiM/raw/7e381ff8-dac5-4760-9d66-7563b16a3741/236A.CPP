#include <algorithm>
#include <iostream>
#include <string>

int main() {
  std::string s;
  std::cin >> s;
  std::sort(s.begin(), s.end());
  s.erase(unique(s.begin(), s.end()), s.end());
  if (s.length() % 2 == 0) {
    std::cout << "CHAT WITH HER!" << std::endl;
  } else
    std::cout << "IGNORE HIM!" << std::endl;
  return 0;
}
