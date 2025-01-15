#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n, a, b, sum(0);
  cin >> n;
  vector<int> v(n);

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    sum += b - a;
    v[i] = sum;
  }
  sort(v.rbegin(), v.rend());
  cout << v.at(0) << "\n";
}
