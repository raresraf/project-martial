#include <bits/stdc++.h>
using namespace std;
int main() {
  int sum = 0;
  int sum1 = 0;
  int n;
  cin >> n;
  int a[100];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum = sum + a[i];
  }
  sort(a, a + n);
  int cnt = 0;
  for (int i = n - 1; i >= 0; i--) {
    if (sum1 <= sum) {
      cnt++;
    } else {
      break;
    }
    sum1 = sum1 + a[i];
    sum = sum - a[i];
  }
  cout << cnt;
}
