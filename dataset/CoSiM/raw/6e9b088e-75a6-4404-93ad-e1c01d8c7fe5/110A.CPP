#include <iostream>
#include <string>

int main() {
  int counter = 0;
  std::string input;

  std::cin >> input;

  for (int i = 0; i < input.length(); i++) {
    if ('7' == input[i] || '4' == input[i])
      counter++;
  }

  std::cout << (7 == counter || 4 == counter ? "YES" : "NO") << std::endl;

  return 0;
}
