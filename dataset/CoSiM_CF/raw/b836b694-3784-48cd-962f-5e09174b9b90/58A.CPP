#include <bits/stdc++.h>
using namespace std;
main() {
  string s, a = "hello";
  int t = 0;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (t == 5)
      break;

    else if (s[i] == a[t])
      t++;
  }
  if (t == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
