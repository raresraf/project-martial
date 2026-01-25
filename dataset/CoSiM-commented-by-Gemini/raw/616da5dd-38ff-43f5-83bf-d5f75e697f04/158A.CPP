#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  int x = a[k - 1];
  if (x > 0) {
    int i = k;
    int cnt = 0;
    while (a[i] == x && i < n) {
      i++;
      cnt++;
    }
    cout << k + cnt << "\n";
  } else {
    int i = 0;
    int cnt = 0;
    while (a[i] != 0 && i < k) {
      i++;
      cnt++;
    }
    cout << cnt << "\n";
  }
  return 0;
}