#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, arr[101], sum = 0, cnt = 0, x = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n);
  for (int i = n - 1; i >= 0; i--) {
    cnt++;
    x += arr[i];
    if (sum / 2 < x)
      break;
  }
  cout << cnt << endl;
  return 0;
}
