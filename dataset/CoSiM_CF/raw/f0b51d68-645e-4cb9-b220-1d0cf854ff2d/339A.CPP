#include <bits/stdc++.h>
using namespace std;

int main() {
  vector<int> v;
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] >= '1' && s[i] <= '3')
      v.push_back(s[i] - '0');
  }
  if (s.size() == 1)
    cout << s << endl;
  else {
    sort(v.begin(), v.end());
    for (int i = 0; i < v.size() - 1; i++) {
      cout << v[i] << "+";
    }
    cout << v[v.size() - 1] << endl;
  }
  return 0;
}
