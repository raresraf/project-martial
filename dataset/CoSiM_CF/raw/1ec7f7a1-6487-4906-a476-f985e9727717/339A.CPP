#include <bits/stdc++.h>
using namespace std;

int main() {
  vector<char> v;
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1' || s[i] == '2' || s[i] == '3') {
      v.push_back(s[i]);
    }
  }
  sort(v.begin(), v.end());
  for (int i = 0; i < v.size(); i++) {
    cout << v[i];
    if (i != v.size() - 1)
      cout << '+';
  }
  cout << endl;
}
