#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> v, d, r;
  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    v.push_back(x);
  }
  int sum = 0;
  sort(v.begin(), v.end());
  d.push_back(-1);
  r.push_back(-1);
  for (int i = 0; i < v.size(); i++) {
    sum += v[i];
    d.push_back(sum);
  }
  sum = 0;
  reverse(v.begin(), v.end());
  for (int i = 0; i < v.size(); i++) {
    sum += v[i];
    r.push_back(sum);
  }
  int i, x = 0;

  if (n % 2) {
    x = 1;
  }

  for (i = 1; i <= n / 2 + x; i++) {
    if (r[i] > d[n - i]) {
      break;
    }
  }
  cout << i << endl;
}