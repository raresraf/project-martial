#include <algorithm>
#include <bits/stdc++.h>
using namespace std;
bool inc(int i, int j) { return (i < j); }
int main() {
  string s;
  cin >> s;
  vector<char> v;
  int n = s.length();
  for (int i = 0; i < n; i++) {
    if (isdigit(s[i])) {
      // cout<<s[i]<<endl;
      v.push_back(s[i]);
    }
  }
  sort(v.begin(), v.end());
  int size_v = v.size();
  for (int i = 0; i < size_v; i++) {

    if (i == size_v - 1)
      cout << v[i];
    else
      cout << v[i] << "+";
  }

  return 0;
}
