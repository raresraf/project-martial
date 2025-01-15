#include <bits/stdc++.h>
#define lli long long int
using namespace std;

int main() {
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    lli sum = 0;
    lli neg = LONG_MIN;
    lli pos = LONG_MIN;
    lli a[n];
    for (int i = 0; i < n; i++)
      cin >> a[i];

    for (int i = 0; i < n; i++) {
      if (a[i] < 0 && a[i] > neg)
        neg = a[i];
      if (a[i] > 0 && a[i] > pos)
        pos = a[i];

      if (a[i + 1] > 0 && i < n - 1 && a[i] < 0) {
        sum += neg;
        // cout<<neg<<endl;
        // cout<<sum<<endl;
        neg = LONG_MIN;

      } else if (a[i + 1] < 0 && i < n - 1 && a[i] > 0) {
        sum += pos;
        // cout<<pos<<endl;
        // cout<<sum<<endl;
        pos = LONG_MIN;
      }
      if (i == n - 1) {
        if (a[i] < 0) {
          if (a[i] > neg)
            neg = a[i];
          sum += neg;
          // cout<<a[i]<<endl;
        } else {
          if (a[i] > pos)
            pos = a[i];
          sum += pos;
          // cout<<a[i]<<endl;
        }
      }
    }
    cout << sum << endl;
  }

  return 0;
}
