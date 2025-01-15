#include <bits/stdc++.h>
using namespace std;

int main() {
  int arr[5][5], x = 0, y = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }
  cout << abs(x - 2) + abs(y - 2);
  return 0;
}
