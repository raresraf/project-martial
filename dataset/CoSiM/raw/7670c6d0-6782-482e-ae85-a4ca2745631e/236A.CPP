#include <bits/stdc++.h>
using namespace std;
int main() {
  map<char, int> mp;
  string s;
  cin >> s;
  int l, cnt = 0;
  l = s.length();
  for (int i = 0; i < l; i++) {
    mp[s[i]]++;
    if (mp[s[i]] == 1) {
      cnt++;
    }
  }
  if (cnt % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}
