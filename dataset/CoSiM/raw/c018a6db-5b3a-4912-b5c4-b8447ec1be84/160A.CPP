#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, m = 0, p = 0, i = 0;
  cin >> n;
  int a[n + 1];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    m = m + a[i];
  }
  sort(a, a + n, greater<int>());

  for (i = 0; i < n; i++) {

    if (p > m - p)
      break;
    else
      p = p + a[i];
  }
  // cout<<p<<endl;
  cout << i << endl;
}