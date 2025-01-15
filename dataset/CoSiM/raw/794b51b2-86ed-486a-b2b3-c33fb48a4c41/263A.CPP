#include <iostream>
using namespace std;
int main() {
  int a[5][5], i, j, pi, pj;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        pi = i;
        pj = j;
      }
    }
  }
  if (pi > 2)
    i = pi - 2;
  else
    i = 2 - pi;
  if (pj > 2)
    j = pj - 2;
  else
    j = 2 - pj;
  cout << (i + j);
  return 0;
}