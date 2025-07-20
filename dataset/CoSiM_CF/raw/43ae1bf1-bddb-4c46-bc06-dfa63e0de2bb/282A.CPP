#include <iostream>

int main() {
  int x = 0;
  int s_count;
  std::cin >> s_count;
  for (int i = 0; i < s_count; i++) {
    std::string temp;
    std::cin >> temp;
    if (temp.find('+') != std::string::npos)
      x++;
    else
      x--;
  }
  std::cout << x << std::endl;
  return 0;
}
