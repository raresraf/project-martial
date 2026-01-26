#include <bits/stdc++.h>
using namespace std;
main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << "\n";

    }

    else {
      cout << s << "\n";
    }
  }
}
