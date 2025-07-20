#include <iostream>
#include <string>

/*
 * Contest: Code Forces Round #163 (Task 266-B)
 * URL: http://codeforces.ru/contest/266/problem/B
 */

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(0);
  std::string s;

  int n, t;

  std::cin >> n >> t >> s;

  while (t--) {
    for (auto it = s.begin(); it != s.end(); ++it) {
      if (*it == 'B' && *(it + 1) == 'G') {
        std::swap(*it, *(it + 1));
        ++it;
      }
    }
  }

  std::cout << s << std::endl;
  return 0;
}
