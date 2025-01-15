#include <bits/stdc++.h>
#include <string>
using namespace std;
// char b[10010];

int main() {
  int n, i = 0, j, u = 0, l, f = 0, min = 1200, max = 0, firstx, lastn, m,
         flag = 0;
  char a[10010], b[10010];
  double x, y, z;

  for (i = 1; i < 6; i++) {
    for (j = 1; j < 6; j++) {
      cin >> n;
      if (n == 1) {
        l = i;
        f = j;
      }
    }
  }
  m = abs(3 - l) + abs(3 - f);
  cout << m << endl;
  return 0;
}