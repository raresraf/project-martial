#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, l;
  string s;
  cin >> n;
  for (int i = 0; i <= n; i++) {
    getline(cin, s);
    l = s.length();
    if (l <= 10)
      cout << s << endl;
    else
      cout << s[0] << l - 2 << s[l - 1] << endl;
  }
}
