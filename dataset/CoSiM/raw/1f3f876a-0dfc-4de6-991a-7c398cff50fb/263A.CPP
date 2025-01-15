#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int arr[5][5], i, j;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] != 0) {
        cout << abs(i - 2) + abs(j - 2);
        return 0;
      }
    }
  }
}
