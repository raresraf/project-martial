#include <bits/stdc++.h>
using namespace std;
int main() {
  int arr[5][5], i, j, xc, yc;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        xc = i;
        yc = j;
      }
    }
  }
  cout << abs(xc - 2) + abs(yc - 2) << endl;
  return 0;
}