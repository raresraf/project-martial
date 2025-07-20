#include <bits/stdc++.h>
using namespace std;

int main() {
  int n = 5;
  int moves = 0;
  while (n--) {
    int a, b, c, d, e;
    cin >> a >> b >> c >> d >> e;
    if (a == 1) {
      moves += 2;
      break;
    }
    if (b == 1) {
      moves += 1;
      break;
    }
    if (c == 1) {
      moves += 0;
      break;
    }
    if (d == 1) {
      moves += 1;
      break;
    }
    if (e == 1) {
      moves += 2;
      break;
    }
  }
  if (n == 4 || n == 0) {
    moves += 2;
  }
  if (n == 3 || n == 1) {
    moves += 1;
  }
  if (n == 2) {
    moves += 0;
  }
  cout << moves << endl;
  return 0;
}
