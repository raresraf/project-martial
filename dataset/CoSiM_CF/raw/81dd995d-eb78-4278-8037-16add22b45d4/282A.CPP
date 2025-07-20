#include <iostream>

int main() {
  int n;
  std::cin >> n;
  std::cin.get();
  int x = 0;

  for (int i = 0; i < n; i++) {
    std::string polecenie;
    std::getline(std::cin, polecenie);
    if (polecenie == "++X" || polecenie == "X++") {
      x++;
    } else if (polecenie == "--X" || polecenie == "X--") {
      x--;
    }
  }

  std::cout << x;

  return 0;
}