// https://codeforces.com/problemset/problem/263/A
#include <bits/stdc++.h>
using namespace std;
int main() {
  int x, y;
  int a[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }
  cout << (abs(x - 2) + abs(y - 2)) << endl;
}
