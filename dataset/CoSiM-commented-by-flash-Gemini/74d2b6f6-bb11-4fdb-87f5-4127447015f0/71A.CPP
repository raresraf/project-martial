#include <bits/stdc++.h>
using namespace std;
int main() {
  int t;
  string s;
  cin >> t;
  while (t--) {
    cin >> s;
    if (s.size() > 10) {
      cout << s[0] << s.size() - 2 << s.back() << endl;
    } else
      cout << s << endl;
  }
  return 0;
}