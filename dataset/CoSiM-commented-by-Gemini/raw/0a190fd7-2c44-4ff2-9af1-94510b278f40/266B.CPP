#include <bits/stdc++.h>

using namespace std;

string solve(int n, int t, string s) {
  while (t--) {
    int i = 0;
    string s2 = "";
    while (i < s.length()) {
      if (i + 1 < s.length() && s[i] == 'B' && s[i + 1] == 'G') {
        s2 += "GB";
        i += 2;
      } else {
        s2 += s[i];
        i++;
      }
    }
    s = s2;
  }

  return s;
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  cout << solve(n, t, s) << endl;
}
