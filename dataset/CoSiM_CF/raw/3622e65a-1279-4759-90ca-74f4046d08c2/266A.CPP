#include <iostream>

int main() {
  int n, count = 0;
  std::string s;
  std::cin >> n;
  std::cin >> s;
  for (int i = 0; i < n - 1; i++) {
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  std::cout << count << std::endl;

  return 0;
}
