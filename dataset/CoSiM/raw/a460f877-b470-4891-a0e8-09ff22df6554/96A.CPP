#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int c = 1, co = 1;
  for (int u = 1; u < s.size(); u++) {
    if (s[u] == s[u - 1]) {
      co++;
    } else {
      if (c < co)
        c = co;
      co = 1;
    }
  }
  if (c < co)
    c = co;
  if (c >= 7) {
    cout << "YES";
    return 0;
  }
  cout << "NO";
}
