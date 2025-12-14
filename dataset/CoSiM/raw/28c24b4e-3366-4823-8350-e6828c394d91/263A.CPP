#include <bits/stdc++.h>

using namespace std;
int main() {
  int arr[5][5];
  int a, b;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        a = i;
        b = j;
      }
    }
  }
  int ans = abs(a - 2) + abs(b - 2);
  cout << ans << endl;
}