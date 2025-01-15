#include <iostream>

/*
 * Contest: Code Forces Round #143 (Task 231-A)
 * URL: http://codeforces.ru/contest/231/problem/A
 */

struct Task {
  int a, b, c;
  Task() {}

  Task(int _a, int _b, int _c) {
    a = _a;
    b = _b;
    c = _c;
  }

  bool solvable(void) { return ((a + b + c) >= 2) ? 1 : 0; }
};

int main(int argc, char **argv) {
  int n, count = 0;
  Task t;
  std::cin >> n;

  while (n--) {
    std::cin >> t.a >> t.b >> t.c;
    if (t.solvable())
      count++;
  }

  std::cout << count << std::endl;
  return 0;
}
