#include <bits/stdc++.h>

using namespace std;

int main() {
  int arr[6][6], fi, fj;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        fi = i;
        fj = j;
      }
    }
  }
  cout << abs(3 - fi) + abs(3 - fj);
  return 0;
}
