// link https://codeforces.com/problemset/problem/59/A

#include <bits/stdc++.h>
using namespace std;

string s;

main() {

  cin >> s;

  if (s[0] > 'Z')
    s[0] -= ' ';

  cout << s;
}
