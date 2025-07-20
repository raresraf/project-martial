#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] < 97) {
      a[i] += 32;
    }
    if (b[i] < 97) {
      b[i] += 32;
    }
  }
  if (a < b) {
    cout << -1 << endl;
  } else if (a > b) {
    cout << 1 << endl;
  } else {
    cout << 0 << endl;
  }
  return 0;
}
