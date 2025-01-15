#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int t, i;
  cin >> t;
  for (i = 0; i < t; i++) {
    cin >> s;
    if (s.length() <= 10)
      cout << s << endl;
    else
      cout << s.at(0) << s.length() - 2 << s.at(s.length() - 1) << endl;
  }
  return 0;
}
// lol
