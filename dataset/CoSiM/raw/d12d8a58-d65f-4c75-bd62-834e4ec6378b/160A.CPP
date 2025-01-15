// Twins
#include <bits/stdc++.h>

using namespace std;
int main() {
  int n, i, sum = 0, sum1 = 0, e = 0, sum_com = 0;
  cin >> n;
  int arr[n];
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    sum = sum + arr[i];
  }
  sort(arr, arr + n);

  for (i = n - 1; i >= 0; i--) {
    sum1 = sum1 + arr[i];
    sum = sum - sum1;
    e++;
    sum_com = sum_com + sum1;
    sum1 = 0;
    if (sum_com > sum)
      break;
  }
  cout << e;

  return 0;
}
