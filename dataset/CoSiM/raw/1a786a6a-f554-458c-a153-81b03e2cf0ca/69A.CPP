#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  int idx = 0;
  int res[n];
  memset(res, 0, sizeof(res));
  for (int i = 1; i < n; i++) {
    if (arr[idx] > arr[i]) {
      res[idx]++;
    } else {
      idx = i;
      res[idx]++;
    }
  }
  for (int i = 0; i < n; i++) {
    cout << res[i] << " ";
  }
  cout << endl;
}