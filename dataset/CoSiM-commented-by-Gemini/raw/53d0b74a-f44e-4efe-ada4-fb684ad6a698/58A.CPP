#include <iostream>
#include <regex>
#include <string>

int main() {
  std::string s;
  std::regex e("^[a-z]*h[a-z]*e[a-z]*l[a-z]*l[a-z]*o[a-z]*$");

  std::cin >> s;

  std::cout << (std::regex_match(s, e) ? "YES" : "NO") << std::endl;

  return 0;
}
