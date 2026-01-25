// link : https://codeforces.com/contest/112/problem/A

#include <bits/stdc++.h>

using namespace std;

string s1, s2;

int k;

main() {

  cin >> s1 >> s2;

  for (int i = 0; i < s1.size(); i++) {

    if (s1[i] < 'a')
      s1[i] += ' ';

    if (s2[i] < 'a')
      s2[i] += ' ';

    if (s1[i] < s2[i]) {
      cout << "-1";
      return 0;
    }
    if (s1[i] > s2[i]) {
      cout << "1";
      return 0;
    }
  }

  cout << "0";
}
