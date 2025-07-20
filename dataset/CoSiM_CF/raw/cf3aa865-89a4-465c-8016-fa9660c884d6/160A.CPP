#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  int s = 0;
  cin >> n;

  // 2cout<<"l";
  vector<int> v(n);
  for (int i = 0; i < n; i++) {
    cin >> v[i];
  }
  sort(v.begin(), v.end());
  // cout<<"j";
  int sum = accumulate(v.begin(), v.end(), 0);
  int t = sum / 2;
  // cout<<sum;
  // cout<<t;
  // int s=0;

  // cout<<k;

  for (k = n - 1; k >= 0; k--) { // cout<<"p";
    s = s + v[k];
    // cout<<s;
    if (s > t) {
      cout << n - k;
      return 0;
    }
  }
}
