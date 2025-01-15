#include <bits/stdc++.h>
using namespace std;

int arr[101];
int main() {
  int n, sum = 0, sum1 = 0, sum2 = 0, flag = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n);
  for (int i = 0; i < n; i++) {

    sum2 = sum - arr[n - i - 1] - sum1;
    sum1 += arr[n - 1 - i];

    if (sum1 > sum2) {
      cout << i + 1;
      flag = 1;

      break;
    }
  }
  if (flag == 0)
    cout << n;
}
