#include <iostream>
#include <vector>

/*
 * Contest: Code Forces Round #350 (Task 670-A)
 * URL: http://codeforces.ru/contest/670/problem/A
 */

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(false);

  int n, min = 0, max = 0;

  std::cin >> n;

  for (int i = 1; i <= n; ++i) {
    if (((i % 7) == 1) || ((i % 7) == 2)) {
      max++;
    }
    if (((i % 7) == 6) || ((i % 7) == 0)) {
      min++;
    }
  }

  std::cout << min << " " << max << std::endl;

  return 0;
}
