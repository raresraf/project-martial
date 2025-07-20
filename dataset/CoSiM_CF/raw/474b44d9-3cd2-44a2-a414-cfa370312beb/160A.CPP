#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

void solve() {
  int n;
  cin >> n;
  vector<int> v;
  int dem = 0, s = 0;
  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    v.push_back(x);
    s += x;
  }
  sort(v.begin(), v.end());
  int ss = 0;
  while (ss <= s) {
    int i = v.back();
    ss += i;
    v.pop_back();
    s -= i;
    dem++;
  }
  cout << dem;
}
int main() {
  solve();

  return 0;
}
