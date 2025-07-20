#include <iostream>
#include <string>

/*
 * Contest: Codeforces Round #89 (Task 118-A)
 * URL: http://codeforces.ru/contest/118/problem/A
 */

int main(int argc, char **argv) {
  std::cin.sync_with_stdio(false);
  std::string s, result = "";

  std::cin >> s;

  for (auto it = s.begin(); it != s.end(); ++it) {
    switch (std::tolower(*it)) {
    case 'a':
    case 'o':
    case 'y':
    case 'e':
    case 'u':
    case 'i':
      break;
    default:
      result.push_back('.');
      result.push_back(static_cast<char>(std::tolower(*it)));
      break;
    }
  }

  std::cout << result << std::endl;
  return 0;
}
