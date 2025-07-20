#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);

  int n;
  std::cin >> n;

  std::string str;
  std::cin >> str;

  char prev_colour = str[0];
  int count = 0;
  for (int i = 1; i < n; i++) {
    if (str[i] == prev_colour)
      count++;
    else
      prev_colour = str[i];
  }
  std::cout << count << std::endl;
  return 0;
}