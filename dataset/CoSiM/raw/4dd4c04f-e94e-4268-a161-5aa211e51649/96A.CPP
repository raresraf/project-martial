#include <cstring>
#include <iostream>
int main() {
  char input[100];
  std::cin.getline(input, 100);
  int amountInARow = 1;
  for (unsigned int i = 1; i < std::strlen(input); i++) {
    if (input[i] == input[i - 1]) {
      amountInARow++;
    } else {
      amountInARow = 1;
    }
    if (amountInARow == 7) {
      std::cout << "YES"
                << "\n";
      return 0;
    }
  }
  std::cout << "NO"
            << "\n";
}
