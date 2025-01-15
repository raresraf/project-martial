#include <iostream>
using namespace std;
int main() {
  int i, j, n, m, o, p;
  int a[1001][4];
  cin >> n;
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  p = 0;
  for (i = 0; i < n; i++) {
    o = 0;
    for (j = 0; j < 3; j++) {
      if (a[i][j] == 1) {
        o++;
      }
    }
    if (o > 1) {
      p++;
    }
  }
  cout << p << endl;
  return 0;
}
