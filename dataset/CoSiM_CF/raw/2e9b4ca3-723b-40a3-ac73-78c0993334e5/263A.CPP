#include <bits/stdc++.h>
using namespace std;

int main() {

  int array[6][6];
  int r, c;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> array[i][j];
      if (array[i][j] > 0) {
        r = i;
        c = j;
      }
    }
  }
  r -= 3;
  c -= 3;
  cout << abs(r) + abs(c) << endl;

  return 0;
}
