#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int i, j, m = 0, pr, pc;
  int a[5][5];
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        pr = i;
        pc = j;
      }
    }
  }
  m = abs(pr - 2) + abs(pc - 2);
  cout << m;
  return 0;
}
