#include <iostream>
int main() {
  std::string s;
  std::cin >> s;
  auto count = 1;
  auto temp = ' ';
  for (auto letter : s) {
    if (temp == letter) {
      count++;
    } else {
      count = 1;
    }
    temp = letter;
    if (count == 7) {
      break;
    }
  }
  if (count == 7) {
    std::cout << "YES" << std::endl;
  } else {
    std::cout << "NO" << std::endl;
  }
  return 0;
}
