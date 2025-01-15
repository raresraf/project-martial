#include <iostream>
#include <string>

int main() {
  std::string s;
  std::string h = "hello";
  std::cin >> s;
  unsigned j = 0;
  for (unsigned i = 0; i < s.size(); i++) {
    if (s[i] == h[j])
      j++;
    if (j == 5) {
      std::cout << "YES\n";
      return 0;
    }
  }
  std::cout << "NO\n";
}
