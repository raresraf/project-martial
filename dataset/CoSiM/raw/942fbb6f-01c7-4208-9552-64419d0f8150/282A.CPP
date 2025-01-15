#include <iostream>

int value(std::string s) { // "++X" or "X++" return 1, "--X" or "X--" return  -1
  if (s[1] == '+')
    return 1;
  else if (s[1] == '-')
    return -1;
}

int main() {
  int n;
  std::cin >> n;
  int count = 0;
  std::string s;
  while (n--) {
    std::cin >> s;
    count += value(s);
  }
  std::cout << count << '\n';
}
