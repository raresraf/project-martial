#include <iostream>

/*
 * Contest: Code Forces Round #304 (Task 546-A)
 * URL: http://codeforces.ru/contest/546/problem/A
 */

int main(int argc, char **argv) {
  int k, n, w, r;
  std::cin >> k >> n >> w;

  r = (k * (w * (w + 1) / 2)) - n;

  std::cout << ((r >= 0) ? r : 0) << std::endl;
  return 0;
}
