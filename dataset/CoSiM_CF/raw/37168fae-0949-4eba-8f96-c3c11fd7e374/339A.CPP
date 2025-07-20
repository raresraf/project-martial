#include <iostream>
#include <sstream>

int main() {
  std::ios::sync_with_stdio(false);
  std::string str;
  std::cin >> str;

  int constanta[3] = {0, 0, 0};
  int idx;
  for (int i = 0; i < str.length(); i += 2) {
    constanta[str[i] - '1']++;
  }

  std::ostringstream res;
  for (int i = 0, j = 0; i < str.length(); i += 2) {
    while (constanta[j] == 0)
      j++;
    if (i == str.length() - 1) {
      res << j + 1;
      break;
    }

    res << j + 1 << "+";
    constanta[j]--;
  }
  std::cout << res.str() << std::endl;

  return 0;
}