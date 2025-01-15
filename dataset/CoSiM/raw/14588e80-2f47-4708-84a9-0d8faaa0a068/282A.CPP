#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n, x = 0;
  cin >> n;
  string s;
  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s[0] == '+' || s[2] == '+') {
      x++;
    } else {
      x--;
    }
  }
  cout << x << endl;

  return 0;
}
