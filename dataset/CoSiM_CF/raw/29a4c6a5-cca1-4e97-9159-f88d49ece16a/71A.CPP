#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    int l = s.length();
    if (l > 10)
      cout << s[0] << l - 2 << s[l - 1] << "\n";
    else
      cout << s << "\n";
  }
  return 0;
}