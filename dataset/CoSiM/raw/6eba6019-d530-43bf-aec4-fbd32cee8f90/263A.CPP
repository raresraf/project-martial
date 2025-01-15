#include <bits/stdc++.h>
using namespace std;

int main() {
  int n = 5;
  int num[n][n];
  int a = 0, b = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cin >> num[i][j];
      if (num[i][j] == 1) {
        a = i + 1;
        b = j + 1;
      }
    }
  }
  if (a <= 3)
    a = 3 - a;
  else
    a -= 3;
  if (b <= 3)
    b = 3 - b;
  else
    b -= 3;
  cout << a + b << endl;

  return 0;
}
