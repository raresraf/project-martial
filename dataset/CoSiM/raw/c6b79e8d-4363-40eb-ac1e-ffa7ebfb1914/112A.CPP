#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  std::string str1, str2;
  std::cin >> str1;
  std::cin >> str2;

  int res = 0;
  int char_code1, char_code2;
  for (int i = 0; i < str1.length(); i++) {
    char_code1 = tolower(str1[i]);
    char_code2 = tolower(str2[i]);
    if (char_code1 < char_code2) {
      res = -1;
      break;
    } else if (char_code1 > char_code2) {
      res = 1;
      break;
    }
  }
  std::cout << res << std::endl;

  return 0;
}