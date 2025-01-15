#include <bits/stdc++.h>
using namespace std;
int main() {
  int a[5][5];
  int i, j, k, l;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      if (a[i][j] == 1) {
        k = i + 1;
        l = j + 1;
        break;
      }
    }
  }
  cout << ((abs(3 - k)) + (abs(3 - l))) << endl;
  return 0;
}
