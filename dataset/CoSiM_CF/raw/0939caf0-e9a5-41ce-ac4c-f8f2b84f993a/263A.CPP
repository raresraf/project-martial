#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int ara[6][6], i, j, m, n;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> ara[i][j];
      if (ara[i][j] == 1) {
        m = i + 1;
        n = j + 1;
      }
    }
  }
  cout << abs(m - 3) + abs(n - 3) << endl;
  return 0;
}
