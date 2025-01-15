#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s.length() < 11)
      cout << s << endl;
    else
      cout << s[0] << s.length() - 2 << s.back() << endl;
  }

  return 0;
}