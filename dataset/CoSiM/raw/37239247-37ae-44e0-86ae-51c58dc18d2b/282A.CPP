// link : https://codeforces.com/problemset/problem/282/A

#include <bits/stdc++.h>

using namespace std;

string s;

int t, k;

main() {

  cin >> t;

  while (t--) {

    cin >> s;

    if (s[1] == '-')
      k--;

    else
      k++;
  }

  cout << k;
}
