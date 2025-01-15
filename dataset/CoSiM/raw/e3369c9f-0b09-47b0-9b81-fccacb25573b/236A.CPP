#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  int ct = 0;
  cin >> s;
  sort(s.begin(), s.end());
  for (int i = 0; i < s.length(); i++) {
    if (s[i] != s[i + 1])
      ct++;
  }
  if (ct % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
