#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, output = 0; // n cases
  cin >> n;
  int matx[n][3];
  for (int i = 0; i < n; i++) {
    int cnt = 0;
    for (int j = 0; j < 3; j++) {
      cin >> matx[i][j];
      if (matx[i][j] == 1)
        cnt++;
    }
    if (cnt >= 2)
      output++;
  }
  cout << output;
  return 0;
}
