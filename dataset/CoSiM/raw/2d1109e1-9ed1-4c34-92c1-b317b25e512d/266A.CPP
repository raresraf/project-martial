#include <iostream>
#include <string>

int main() {
  unsigned deleteStones = 0;
  unsigned stoneNumber = 0;
  std::string input;

  char currentColor, nextColor;

  std::cin >> stoneNumber;
  std::cin >> input;

  for (int i = 0; i < stoneNumber - 1; i++) {
    currentColor = input[i];
    nextColor = input[i + 1];

    if (currentColor == nextColor) {
      deleteStones++;
    }
  }

  std::cout << deleteStones << std::endl;

  return 0;
}
