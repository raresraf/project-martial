#include <bits/stdc++.h>
#define int long long int

using namespace std;

void solve() {
  int n;
  cin >> n;
  vector<int> arr(n);
  int resPos = 0, resNeg = 0, tempSum = 0, maxSum = 0;
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    arr[i] = temp;
  }
  for (int i = 0; i < n;) {
    if (arr[i] > 0) {
      resPos = arr[i];
      while (i < n && arr[i] > 0) {
        resPos = max(resPos, arr[i]);
        i++;
      }
    } else {
      resNeg = arr[i];
      while (i < n && arr[i] < 0) {
        resNeg = max(resNeg, arr[i]);
        i++;
      }
    }
    tempSum = resPos + resNeg;
    resPos = 0;
    resNeg = 0;
    maxSum += tempSum;
  }
  cout << maxSum << endl;
}

int32_t main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int t;
  cin >> t;
  while (t--)
    solve();
  return 0;
}