#include <iostream>
#include <set>

/*
 * Contest: Code Forces Round #146 (Task 236-A)
 * URL: http://codeforces.ru/contest/236/problem/A
 */

int main(int argc, char **argv) {
  std::set<int> ch;

  std::string s;
  std::cin >> s;

  for (auto it = s.cbegin(); it != s.cend(); ++it) {
    ch.insert(*it);
  }

  std::cout << ((ch.size() & 1) ? "IGNORE HIM!" : "CHAT WITH HER!")
            << std::endl;
  return 0;
}
