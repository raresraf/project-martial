#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  int i, j, c = 0, d;
  for (i = 0; i < s.size(); i++) {
    d = 1;
    for (j = 0; j < i; j++) {
      if (s[i] == s[j])
        d = 0;
    }
    c = c + d;
  }

  if (c % 2 == 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
