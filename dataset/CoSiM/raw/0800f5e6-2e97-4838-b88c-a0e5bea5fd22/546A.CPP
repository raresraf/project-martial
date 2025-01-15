#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input", "r", stdin);
  freopen("output", "w", stdout);
#endif
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  cin >> n;
  int arr[n][2];
  for (int i = 0; i < n; i++) {
    cin >> arr[i][0];
    cin >> arr[i][1];
  }

  int k;
  cin >> k;

  for (int i = 0; i < n; i++) {
    if (k <= arr[i][1]) {
      cout << n - i << endl;
      return 0;
    }
  }
}
