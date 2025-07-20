#include <bits/stdc++.h>
using namespace std;

int main() {

  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    int n = s.length();
    if (n > 10) {
      char f = s[0];
      char l = s[n - 1];
      cout << f << n - 2 << l << endl;
    } else
      cout << s << endl;
  }

  return 0;
}
