#include <iostream>

int main() {
  std::string druzyna;
  std::cin >> druzyna;
  int dlugosc, ile = 0;
  dlugosc = druzyna.length();

  for (int i = 1; i < dlugosc + 1; i++) {
    if (druzyna[i] == druzyna[i - 1]) {
      ile++;
    } else if (dlugosc - 1 == i) {
      if (druzyna[dlugosc] == druzyna[i]) {
        ile++;
      }
    } else {
      ile++;
      if (ile >= 7) {
        std::cout << "YES";
        break;
      } else {
        ile = 0;
      }
    }

    if (ile >= 7) {
      std::cout << "YES";
      break;
    }
  }

  if (ile < 7) {
    std::cout << "NO";
  }
  return 0;
}