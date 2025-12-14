#include <iostream>

using std::cin;
using std::cout;

int main(void) {
  int n;
  int sures[] = {0, 0, 0};

  cin >> n;
  int count = 0;
  int solutionCount = 0;

  while (count++ < n) {
    cin >> sures[0] >> sures[1] >> sures[2];
    int sureCount = 0;
    for (int i = 0; i < 3; i++) {
      if (sures[i]) {
        sureCount++;
      }
    }
    if (sureCount >= 2) {
      solutionCount++;
    }
  }

  cout << solutionCount;

  return 0;
}