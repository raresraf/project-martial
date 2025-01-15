#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  string s;

  for (int i = 1; i <= n; i++) {
    cin >> s;

    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1];
      cout << endl;
    } else {
      cout << s << endl;
    }
  }

  return 0;
}
