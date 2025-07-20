#include <iostream>
#include <string>

/*
 * Contest: Code Forces Round #84 (Task 110-A)
 * URL: http://codeforces.ru/contest/110/problem/A
 */

bool ishappy(int n) {
  if (n == 0)
    return 0;

  while (n > 0) {
    if ((n % 10 == 4) || (n % 10 == 7))
      n /= 10;
    else
      return false;
  }

  return true;
}

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(0);

  std::string s;
  std::cin >> s;

  int count = 0;

  for (auto it = s.begin(); it != s.end(); ++it) {
    if (*it == '4' || *it == '7')
      count++;
  }

  std::cout << (ishappy(count) ? "YES" : "NO") << std::endl;

  return 0;
}
