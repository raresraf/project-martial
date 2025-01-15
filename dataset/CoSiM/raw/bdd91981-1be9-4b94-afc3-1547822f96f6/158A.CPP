#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int a[100005];
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (int i = 0; i < n; i++) {
    //	cin>>a[i]];
    if (a[i] >= a[k - 1] && a[i])
      cnt++;
  }
  cout << cnt << endl;
}