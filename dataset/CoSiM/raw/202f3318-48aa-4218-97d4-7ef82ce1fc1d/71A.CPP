#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  int n, strLen;
  cin >> n;
  string s;
  for (int i = 0; i < n; ++i) {
    cin >> s;
    strLen = s.length();

    if (strLen > 10) {
      cout << s[0] << strLen - 2 << s[strLen - 1] << "\n";
    } else
      cout << s << "\n";
  }

  return 0;
}
