#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, sum = 0, sum2 = 0;
  cin >> n;
  vector<int> a(n);
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a.begin(), a.end());
  for (int i = n - 1;; i--) {
    sum2 += a[i];
    if (sum2 > sum - sum2) {
      cout << n - i;
      break;
    }
  }
}