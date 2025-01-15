#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  int pass = k;
  int base = -1;

  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;

    if (i <= k - 1) {
      if (i == k - 1)
        base = temp;
      if (temp == 0) {
        pass--;
      } else {
        continue;
      }
    }

    if (temp > 0 && temp >= base)
      pass++;
  }

  cout << pass << endl;

  return 0;
}
