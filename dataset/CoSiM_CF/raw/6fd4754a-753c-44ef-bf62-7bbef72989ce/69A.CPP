// Problem Link https://codeforces.com/problemset/problem/69/A
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int x[n];
  int y[n];
  int z[n];
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }

  int sx = 0, sy = 0, sz = 0;
  for (int i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
    sz += z[i];
  }

  if (sx == 0 && sy == 0 && sz == 0) {
    cout << "YES"
         << "\n";
  } else {
    cout << "NO"
         << "\n";
  }
}