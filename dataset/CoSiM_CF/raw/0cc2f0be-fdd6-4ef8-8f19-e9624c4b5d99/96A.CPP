#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, n;
  string s;
  int c1 = 0, c2 = 0;
  cin >> s;
  for (i = 1; i < s.length(); i++) {
    if (s[i] == s[i - 1])
      c1 += 1;
    else {
      c1 = 0;
    }
    if (c1 == 6) {
      c2 += 1;
      break;
    }
  }
  if (c2 == 0)
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
  return 0;
}
