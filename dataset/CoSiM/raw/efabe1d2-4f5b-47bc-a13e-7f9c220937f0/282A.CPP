#include <iostream>
#include <string>

/*
 * Contest: Code Forces Round #173 (Task 282-A)
 * URL: http://codeforces.ru/contest/282/problem/A
 */

int main(int argc, char **argv) {
  int n, value = 0;
  std::string s;

  std::cin >> n;
  while (n--) {
    std::cin >> s;
    if (s[1] == '+')
      value++;
    else
      value--;
  }

  std::cout << value << std::endl;
  return 0;
}
