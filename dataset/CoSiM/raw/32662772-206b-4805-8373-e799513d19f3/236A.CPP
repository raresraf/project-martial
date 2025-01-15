#include <cstring>
#include <iostream>
#include <set>

int main() {
  int i;
  std::set<char> strset;
  char str[100];
  std::cin >> str;
  for (i = 0; i < strlen(str); i++) {
    strset.insert(str[i]);
  }
  if (strset.size() % 2 == 0)
    std::cout << "CHAT WITH HER!" << '\n';
  else
    std::cout << "IGNORE HIM!" << '\n';
  return 0;
}
