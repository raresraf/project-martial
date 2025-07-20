#include <iostream>
using namespace std;
int main() {
  int a[5][5];
  int i, j, r, c;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        r = i;
        c = j;
      }
    }
  }
  if (r >= 2)
    r = r - 2;
  else
    r = 2 - r;
  if (c >= 2)
    c = c - 2;
  else
    c = 2 - c;
  cout << r + c << endl;
  return 0;
}
