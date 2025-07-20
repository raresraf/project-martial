#include <iostream>
#include <string>

void occ(std::string &s, char c) {
  unsigned oc{};
  for (auto i = 0; i < s.size(); i++) {
    if (s[i] == c)
      oc++;
    if (oc > 1) {
      s[i] = '1';
      oc = 0;
    }
  }
}

unsigned distinctL(std::string s) {
  unsigned count{}, temp{};
  for (auto i = 0; i < s.size(); i++) {
    occ(s, s[i]);
  }
  for (auto i = 0; i < s.size(); i++) {
    (s[i] > 96 && s[i] < 123) ? count++ : count;
  }
  return count;
}

int main() {
  std::string s;
  do {
    std::cin >> s;
  } while (s.size() < 1 || s.size() > 100);
  for (auto i = 0; i < s.size(); i++) {
    s[i] = (s[i] < 91 && s[i] > 64) ? s[i] + 32 : s[i];
  }
  if (distinctL(s) % 2)
    std::cout << "IGNORE HIM!";
  else
    std::cout << "CHAT WITH HER!";
  return 0;
}
