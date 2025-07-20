#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int i, j, k, n;
  cin >> n;
  int ans = 0;
  cin >> s;
  char a = s[0];
  for (i = 1; i < n; i++) {
    if (s[i] == a) {
      ans++;
    } else
      a = s[i];
  }
  cout << ans << endl;
  return 0;
}