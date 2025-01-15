#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  int a[n], x;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    if (i == k - 1)
      x = a[i];
  }
  int count = 0;

  for (int i = 0; i < n; i++) {
    if (a[i] < x)
      break;
    else if (a[i] > 0 && a[i] >= x)
      count++;
  }
  cout << count;

  return 0;
}
