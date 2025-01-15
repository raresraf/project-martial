#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, s2;
  cin >> s >> s2;

  for (int i = 0; i < s.size(); ++i) {
    if (s[i] < 97)
      s[i] += 32;
    if (s2[i] < 97)
      s2[i] += 32;

    if ((int)s[i] < (int)s2[i]) {
      cout << "-1" << endl;
      return 0;
    } else if ((int)s[i] > (int)s2[i]) {
      cout << "1" << endl;
      return 0;
    }
  }

  cout << "0" << endl;
  return 0;
}

// codeforces.com/problemset/problem/112/A

// 'A' = 65
// 'a' = 97
// dif = 32