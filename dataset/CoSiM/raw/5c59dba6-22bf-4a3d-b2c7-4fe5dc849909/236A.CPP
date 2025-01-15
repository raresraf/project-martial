#include <iostream>
#include <set>

int main() {
  std::ios::sync_with_stdio(false);
  std::string name;
  std::cin >> name;
  std::set<char> unique_char;
  for (char i : name) {
    unique_char.insert(i);
  }
  if (unique_char.size() & 1)
    std::cout << "IGNORE HIM!" << std::endl;
  else
    std::cout << "CHAT WITH HER!" << std::endl;
  return 0;
}