#include <bits/stdc++.h>
#include <string>
using namespace std;
int i, t = 0, n, a;
int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n;
  cin >> n;
  int arr[n];
  int sum = 0;
  int counter = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sum = sum / 2;
  sort(arr, arr + n);
  int sum2 = 0;
  for (int i = n - 1; i >= 0; i--) {
    sum2 += arr[i];
    ++counter;
    if (sum2 > sum) {
      break;
    }
  }
  cout << counter;
  return 0;
}
