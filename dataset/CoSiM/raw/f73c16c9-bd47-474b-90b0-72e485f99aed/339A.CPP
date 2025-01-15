#include <bits/stdc++.h>

using namespace std;

int main() {

  string s;

  string help = "";

  string ans = "-";

  cin >> s;

  for (int i = 0; i < s.length(); ++i)
    if (s[i] != '+')
      help += s[i];

  sort(help.begin(), help.end());

  // cout << help << endl;

  ans[0] = help[0];

  for (int i = 1; i < help.length(); ++i) {

    ans += '+';

    ans += help[i];
  }

  cout << ans << endl;
}