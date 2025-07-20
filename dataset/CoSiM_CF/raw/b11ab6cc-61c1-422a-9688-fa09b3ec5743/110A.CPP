#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  string s;
  cin >> s;
  int n = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      n += 1;
    }
  }
  if (n == 4 || n == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
