#include <iostream>
#include <string>

/*
 * Contest: Code Forces Round #163 (Task 266-A)
 * URL: http://codeforces.ru/contest/266/problem/A
 */

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(0);
  std::string s;

  int stones, count = 0;

  std::cin >> stones >> s;

  for (auto it = s.cbegin(); it != s.cend() - 1; ++it) {
    if (*it == *(it + 1))
      count++;
  }

  std::cout << count << std::endl;
  return 0;
}
