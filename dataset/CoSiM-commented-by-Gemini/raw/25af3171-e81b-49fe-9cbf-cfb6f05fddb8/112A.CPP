#include <iostream>
#include <string>

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);
  std::string str1, str2;
  std::cin >> str1 >> str2;
  bool isSame = true;
  for (size_t i = 0; i < str1.length(); i++) {
    char a = toupper(str1[i]), b = toupper(str2[i]);
    if (a != b) {
      isSame = false;
      std::cout << ((a > b) ? "1" : "-1");
      break;
    }
  }
  if (isSame) {
    std::cout << "0";
  }
  return 0;
}