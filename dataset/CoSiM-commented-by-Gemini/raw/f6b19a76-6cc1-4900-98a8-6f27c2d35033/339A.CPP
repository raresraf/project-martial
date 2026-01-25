#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  vector<char> v;

  for (int i = 0; i < int(s.size()); i += 2) {
    v.push_back(s[i]);
  }

  sort(v.begin(), v.end());

  for (int i = 0; i < int(v.size()); i++) {
    if (i != v.size() - 1) {
      cout << v[i] << "+";
    } else {
      cout << v[i];
    }
  }

  return 0;
}
