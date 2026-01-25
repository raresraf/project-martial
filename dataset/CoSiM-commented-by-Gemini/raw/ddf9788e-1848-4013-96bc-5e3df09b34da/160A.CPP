#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int arr[n] = {0};
  int cnt = 0, sum = 0, total = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n, greater<int>());
  for (int i = 0; i < n; i++) {
    if (total + arr[i] <= sum / 2) {
      cnt++;
      total += arr[i];
    } else
      break;
  }
  cout << cnt + 1 << endl;
}
