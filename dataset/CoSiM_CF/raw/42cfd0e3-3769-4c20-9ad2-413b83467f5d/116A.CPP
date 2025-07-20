#include <iostream>

int main() {
  int numStations;
  std::cin >> numStations;
  int currentCount = 0, maxCount = 0;
  while (numStations--) {
    int exit, enter;
    std::cin >> exit >> enter;
    currentCount += enter - exit;
    maxCount = std::max(currentCount, maxCount);
  }
  std::cout << maxCount << '\n';
}
