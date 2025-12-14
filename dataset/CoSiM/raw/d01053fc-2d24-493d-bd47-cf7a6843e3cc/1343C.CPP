#include <bits/stdc++.h>
using namespace std;
#define endl "\n"

int main() {
  ios_base ::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int t, n;
  cin >> t;
  while (t--) {
    cin >> n;
    int a[n];
    for (int i = 0; i < n; i++) {
      cin >> a[i];
    }
    long long sum = 0;
    int i = 0;
    while (i < n) {
      int maxm = INT_MIN;
      while (i < n && a[i] < 0) {
        maxm = max(maxm, a[i]);
        i++;
      }
      if (maxm != INT_MIN) {
        sum += maxm;
      }
      maxm = INT_MIN;
      while (i < n && a[i] > 0) {
        maxm = max(maxm, a[i]);
        i++;
      }
      if (maxm != INT_MIN) {
        sum += maxm;
      }
    }
    cout << sum << endl;
  }

  return 0;
}