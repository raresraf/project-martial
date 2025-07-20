#include <iostream>

/*
 * Contest: Code Forces Round #87 (Task 116-A)
 * URL: http://codeforces.ru/contest/116/problem/A
 */

int main(int argc, char **argv) {
  int n, in, out, max, capacity = 0;
  std::cin >> n;

  while (n--) {
    std::cin >> in >> out;
    capacity += out - in;

    if (max < capacity)
      max = capacity;
  }

  std::cout << max << std::endl;
  return 0;
}
