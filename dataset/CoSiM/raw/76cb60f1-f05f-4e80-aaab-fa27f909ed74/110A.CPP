#include <iostream>

int main() {
  long long int n;
  std::cin >> n;
  int co = 0;
  bool ans = true;
  while (n != 0) {
    if (n % 10 == 4 || n % 10 == 7)
      co++;
    n /= 10;
  }
  if (co == 0)
    ans = false;
  else {
    while (co != 0) {
      if (co % 10 != 4 && co % 10 != 7) {
        ans = false;
      }
      co /= 10;
    }
  }
  (ans) ? std::cout << "YES" << std::endl : std::cout << "NO" << std::endl;
}
