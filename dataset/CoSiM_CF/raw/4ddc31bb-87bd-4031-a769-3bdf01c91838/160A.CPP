#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> a(n);
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
  }
  sort(a.begin(), a.end());
  int minSum = 0, totalSum = accumulate(a.begin(), a.end(), 0);

  for (int i = n - 1; i >= 0; i--) {
    minSum += a[i];
    if (minSum > (totalSum - minSum)) {
      cout << n - i << "\n";
      break;
    }
  }
  return 0;
}