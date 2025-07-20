// link : https://codeforces.com/problemset/problem/71/A

#include <bits/stdc++.h>
using namespace std;

string s;

int t;

main() {

  cin >> t;

  while (t--) {

    cin >> s;

    if (s.size() > 10)
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;

    else
      cout << s << endl;
  }
}
