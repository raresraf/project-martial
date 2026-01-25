// https://codeforces.com/problemset/problem/339/A
// 339A. Helpful Maths
#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  vector<int> v;
  for (int i = 0; i < str.size(); i = i + 2) {
    v.push_back((int)str[i] - 48);
  }
  sort(v.begin(), v.end());
  for (int i = 0; i < v.size() - 1; i++) {
    cout << v[i] << "+";
  }
  cout << v[v.size() - 1];
  return 0;
}
