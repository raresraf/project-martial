#include <bits/stdc++.h>
using namespace std;
int main() {
  int **arr = new int *[5];
  for (int i = 1; i <= 5; i++) {
    arr[i] = new int[5];

    for (int j = 1; j <= 5; j++) {
      cin >> arr[i][j];
    }
  }
  bool isfound = false;
  int i, j;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      if (arr[i][j] == 1) {
        isfound = true;
        break;
      }
    }
    if (isfound) {
      break;
    }
  }
  int sum = abs(3 - i) + abs(3 - j);
  cout << sum << "\n";
}
