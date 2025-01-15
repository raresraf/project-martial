#include <iostream>

int main() {
  unsigned n{}, count{};
  char t;
  do {
    std::cin >> n;
  } while (n < 1 || n > 50);
  char *s = new char[n + 1];
  for (auto i = 0; i < n; i++) {
    std::cin >> t;
    s[i] = t;
  }
  for (auto i = 0; i < n - 1; i++) {
    if (s[i] == s[i + 1])
      count++;
  }
  std::cout << count;
  delete[] s;
  return 0;
}
