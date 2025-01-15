#include <iostream>

int main() {
  int liczbaKul, output = 0;
  std::string kolory;
  std::cin >> liczbaKul;
  std::cin >> kolory;

  for (int i = 1; i < liczbaKul; i++) {
    if (kolory[i] == kolory[i - 1]) {
      output++;
    }
  }

  std::cout << output;
  return 0;
}