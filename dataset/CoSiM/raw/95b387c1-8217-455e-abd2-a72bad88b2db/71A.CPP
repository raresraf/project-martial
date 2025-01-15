#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  int i, n, l;
  cin >> n;
  for (i = 0; i < n; i++) {
    string s;
    cin >> s;
    l = s.size();
    if (l <= 10) {
      cout << s << endl;

    } else
      cout << s[0] << l - 2 << s[l - 1] << endl;
  }
  return 0;
}
