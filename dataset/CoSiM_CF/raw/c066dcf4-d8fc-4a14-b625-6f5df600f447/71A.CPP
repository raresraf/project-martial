#include <bits/stdc++.h>
using namespace std;

void abbreviate(string s) {
  if (s.length() > 10) {
    cout << s[0] << to_string(s.length() - 2) << s[s.length() - 1] << endl;
  } else {
    cout << s << endl;
  }
}

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string w;
    cin >> w;
    abbreviate(w);
  }

  return 0;
}
