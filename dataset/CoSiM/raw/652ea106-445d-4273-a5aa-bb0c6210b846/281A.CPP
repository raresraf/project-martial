#include <iostream>

using namespace std;

int main() {
  string s;
  std::cin >> s;
  if (int(s[0]) > 91)
    // s[0] = putchar(toupper(s[0]));
    s[0] = int(s[0]) - 32;
  std::cout << s;
  return 0;
}