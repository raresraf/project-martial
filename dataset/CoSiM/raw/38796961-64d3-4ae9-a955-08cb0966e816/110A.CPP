// link : https://codeforces.com/problemset/problem/110/A

#include <bits/stdc++.h>
using namespace std;

string s;

main() {

  long long a;
  cin >> a;

  s = to_string(a);

  int k = 0;

  for (int i = 0; i < s.size(); i++) {

    if (s[i] == '4' || s[i] == '7')
      k++;
  }

  if (k == 7 || k == 4)
    cout << "YES" << endl;

  else
    cout << "NO" << endl;
}
