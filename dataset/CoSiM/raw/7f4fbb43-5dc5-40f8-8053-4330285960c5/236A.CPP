#include <bits/stdc++.h>
#include <map>
using namespace std;
void solve() {
  map<char, int> m;
  string s;
  cin >> s;
  int n = s.length(), dem = 0;

  for (int i = 0; i < n; i++) {
    m[s[i]]++;
    if (m[s[i]] == 1) {
      dem++;
    }
  }
  // cout<<dem<<endl;
  if (dem % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}
int main() {
  solve();
  return 0;
}
