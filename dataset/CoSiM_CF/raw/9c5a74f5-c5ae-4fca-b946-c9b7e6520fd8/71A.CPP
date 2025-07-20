#include <iostream>
#include <string>
#include <vector>

int main() {
  unsigned num{};
  std::string s;
  std::vector<std::string> v;
  do {
    std::cin >> num;
  } while (num > 100 || num < 1);
  for (auto i = 0; i < num; i++) {
    do {
      std::cin >> s;
    } while (s.size() > 100 || s.size() < 1);
    v.push_back(s);
  }
  for (auto i = 0; i < num; i++) {
    if (v.at(i).size() > 10) {
      s = v.at(i);
      v.at(i).clear();
      v.at(i) = s[0];
      v.at(i) += std::to_string(s.size() - 2);
      v.at(i) += s[s.size() - 1];
    }
  }
  for (auto i : v)
    std::cout << i << '\n';
  return 0;
}
