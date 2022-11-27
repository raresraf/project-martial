#include <iostream>

int main() {

  int kilos = 0;
  std::cin >> kilos;

  if (kilos % 2 == 0) {
    if (kilos <= 2) {
      std::cout << "NO";
    } else {
      std::cout << "YES";
    }

  } else {
    std::cout << "NO";
  }

  return 0;
}