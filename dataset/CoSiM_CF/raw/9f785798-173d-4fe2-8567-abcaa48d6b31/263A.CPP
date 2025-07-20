#include <bits/stdc++.h>

using namespace std;

int main() {
  int arr[6][6], sum = 0;

  for (int i = 1; i < 6; i++) {
    for (int j = 1; j < 6; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        sum = abs(i - 3) + abs(j - 3);
      }
    }
  }

  cout << sum << endl;
}
