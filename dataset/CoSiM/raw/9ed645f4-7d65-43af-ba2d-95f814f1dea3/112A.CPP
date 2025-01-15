#include <iostream>
#include <string>

/*
 * Contest: Code Forces Round #85 (Task 112-A)
 * URL: http://codeforces.ru/contest/112/problem/A
 */

int inline compare(char a, char b) { return (a < b) ? -1 : (a == b) ? 0 : 1; }

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(0);

  std::string a, b;
  int result;

  std::cin >> a >> b;

  for (int i = 0; i < a.size(); ++i) {
    if (result = compare(std::tolower(a[i]), std::tolower(b[i])))
      break;
  }

  std::cout << result << std::endl;
  return 0;
}
