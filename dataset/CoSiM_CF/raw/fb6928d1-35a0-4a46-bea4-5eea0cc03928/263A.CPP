#include <iostream>

int main() {
  int matrix[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cin >> matrix[i][j];
    }
  }

  int pozycjaI = 0, pozycjaJ = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (matrix[i][j] == 1) {
        pozycjaI = i + 1;
        pozycjaJ = j + 1;
        break;
      }
    }
  }

  int wynik;
  if (pozycjaI == 3 && pozycjaJ == 3) {
    wynik = 0;
    std::cout << wynik;
  } else if (pozycjaI == 3) {
    wynik = abs(3 - pozycjaJ);
    std::cout << wynik;
  } else if (pozycjaJ == 3) {
    wynik = abs(3 - pozycjaI);
    std::cout << wynik;
  } else {
    wynik = abs(3 - pozycjaI) + abs(3 - pozycjaJ);
    std::cout << wynik;
  }

  return 0;
}