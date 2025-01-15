#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int x, y;
  for (int i = 0; i < 5; ++i) {
    bool isFound = false;
    for (int j = 0; j < 5; ++j) {
      int num;
      cin >> num;
      if (num == 1) {
        x = i;
        y = j;
        break;
      }
    }
    if (isFound == true)
      break;
  }
  cout << abs(2 - x) + abs(2 - y);
  return 0;
}
