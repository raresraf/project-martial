#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  int sum = 0;
  int count = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sum = sum / 2;
  sort(arr, arr + n);
  int sum2 = 0;
  for (int i = n - 1; i >= 0; i--) {
    sum2 += arr[i];
    ++count;
    if (sum2 > sum) {
      break;
    }
  }

  cout << count;

  return 0;
}
