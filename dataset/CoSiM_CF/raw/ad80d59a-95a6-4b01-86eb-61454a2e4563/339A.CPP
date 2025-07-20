#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  _ string s;
  int a[3] = {0, 0, 0};
  cin >> s;
  for (auto x : s) {
    if (x <= '3' && x >= '1') {
      a[x - '1']++;
    }
  }
  bool prev = false;
  for (int i = 0; i < a[0]; i++) {
    if (prev)
      cout << '+';
    cout << 1;
    prev = true;
  }
  for (int i = 0; i < a[1]; i++) {
    if (prev)
      cout << '+';
    cout << 2;
    prev = true;
  }
  for (int i = 0; i < a[2]; i++) {
    if (prev)
      cout << '+';
    cout << 3;
    prev = true;
  }

  return 0;
}