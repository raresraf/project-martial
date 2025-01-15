#include <iostream>
#include <string>

/*
 * Contest: Code Forces Round #172 (Task 281-A)
 * URL: http://codeforces.ru/contest/281/problem/A
 */

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(0);

  std::string s;
  std::cin >> s;

  s[0] = std::toupper(s[0]);

  std::cout << s << std::endl;
  return 0;
}
