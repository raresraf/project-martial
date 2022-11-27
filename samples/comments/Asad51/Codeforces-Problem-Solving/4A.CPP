#include <bits/stdc++.h>
using namespace std;

int main() {
  int w, i, j, c = 0;
  cin >> w;
  if (w % 2 == 1) {
    cout << "NO" << endl;
    return 0;
  }
  for (i = 2; i < w; i = i + 2) {
    for (j = i; j < w; j = j + 2) {
      if ((i + j) == w) {
        c = 1;
        break;
      }
    }
  }

  if (c == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
