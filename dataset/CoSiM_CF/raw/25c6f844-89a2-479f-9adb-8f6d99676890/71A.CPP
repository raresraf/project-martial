#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    int i = s.length();
    if (i <= 10)
      cout << s << endl;
    else {
      cout << s[0] << i - 2 << s[i - 1] << endl;
    }
  }
  return 0;
}
