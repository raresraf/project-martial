#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, j = 0, c = 1;
  string p;
  cin >> p;
  for (i = 0; i < p.size() - 1; i++) {
    if (p[i] == p[i + 1]) {
      c++;
    } else {
      c = 1;
    }

    if (c >= 7)
      j = 1;
  }
  if (j == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
