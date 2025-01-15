#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, x = 0, y = 0, z = 0;
  cin >> n;
  int ar[3][n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> ar[j][i];
    }
  }
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < n; b++) {
      if (a == 0) {
        x += ar[a][b];
      }
      if (a == 1) {
        y += ar[a][b];
      }
      if (a == 2) {
        z += ar[a][b];
      }
    }
  }
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
