#include <bits/stdc++.h>
using namespace std;
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
int main()

{
  IOS int n;
  cin >> n;
  int a[n], b[n];
  int i, c = 0, sum = 0;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a, a + n, greater<int>());

  for (i = 1; i < n; i++) {
    a[i] += a[i - 1];
  }
  // cout<<sum<<endl;
  for (i = 0; i < n; i++) {
    // cout<<a[i]<<" "<<endl;
    if (a[i] > sum - a[i]) {
      c++;
      break;
    } else {
      c++;
    }
  }
  cout << c;
}