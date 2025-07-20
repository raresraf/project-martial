#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  vector<int> v;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] != '+')
      v.push_back(s[i] - '0');
  }
  sort(v.begin(), v.end());
  int x = v.at(v.size() - 1);
  v.pop_back();
  for (auto i : v) {
    cout << i << "+";
  }
  cout << x << "\n";
}