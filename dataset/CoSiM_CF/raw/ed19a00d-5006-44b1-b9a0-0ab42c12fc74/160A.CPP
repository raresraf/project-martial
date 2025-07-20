#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, a[100], sum = 0, summ = 0;

  cin >> t;

  for (int i = 0; i < t; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a, a + t, greater<int>());
  sum = sum / 2 + 1;

  for (int i = 0; i < t; i++) {
    summ += a[i];
    if (summ >= sum) {
      cout << i + 1 << endl;
      return 0;
    }
  }
}
