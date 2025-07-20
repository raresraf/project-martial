#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, sum = 0;
  int a[100];
  for (i = 1; i <= 100; i++)
    a[i] = 0;
  cin >> n;
  int arr[n];
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  int div = sum / 2;
  sort(arr, arr + n);
  int s = 0;
  int coins = 0;
  for (i = n - 1; i >= 0; i--) {
    if (s > div)
      break;
    else {
      s += arr[i];
      coins++;
    }
  }
  cout << coins;
  return 0;
}
