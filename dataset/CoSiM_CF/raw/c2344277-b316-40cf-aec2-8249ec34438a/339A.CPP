#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  vector<int> v;
  int i, l;
  for (i = 0; i < s.size(); i += 2) {
    v.push_back(s[i]);
  }
  l = v.size();
  sort(v.begin(), v.end());
  for (i = 0; i < (l - 1); i++) {
    cout << v[i] - 48 << "+";
  }
  cout << v[l - 1] - 48;
}
