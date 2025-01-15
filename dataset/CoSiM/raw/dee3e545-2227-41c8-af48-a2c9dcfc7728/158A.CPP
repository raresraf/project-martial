#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int c = 0;
  int a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0)
      c++;
  }
  cout << c;
  return 0;
}
