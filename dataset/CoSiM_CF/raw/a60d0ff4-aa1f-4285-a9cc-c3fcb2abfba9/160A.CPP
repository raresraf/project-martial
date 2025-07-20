#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, sum = 0, halfsum = 0, cnt = 0;
  cin >> n;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  sort(arr, arr + n);

  for (int i = n - 1; i >= 0; i--) {
    halfsum += arr[i];
    cnt++;
    if (halfsum > sum / 2) {
      cout << cnt;
      break;
    }
  }

  return 0;
}
