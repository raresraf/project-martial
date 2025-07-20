#include <iostream>

int main() {
  int n, x = 0;
  std::string s;
  std::cin >> n;
  for (int i = 1; i <= n; i++) {
    std::cin >> s;
    if (s[1] == '+')
      x++;
    else
      x--;
  }
  std::cout << x;
}
