#include <iostream>

int main() {
  int nOfStatements = 0;
  int x = 0;
  std::cin >> nOfStatements;
  for (int i = 0; i < nOfStatements; i++) {
    std::string aString;
    std::cin >> aString;
    if (aString == "++X" || aString == "X++")
      x++;
    else
      x--;
  }
  std::cout << x << std::endl;
  return 0;
}
