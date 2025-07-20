#include <iostream>
#include <math.h>
using namespace std;

int main() {

  int x, y;
  int ar[10][10];

  for (int i = 1; i <= 5; ++i) {
    for (int j = 1; j <= 5; ++j) {
      cin >> ar[i][j];
      if (ar[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  int ans = abs(x - 3) + abs(y - 3);

  cout << ans;

  return 0;
}
