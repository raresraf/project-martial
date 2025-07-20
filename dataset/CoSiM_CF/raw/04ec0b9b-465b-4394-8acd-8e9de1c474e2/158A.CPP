#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, dem = 0, a[50];
  cin >> n >> k;
  for (int i = 0; i < n; i++)
    cin >> a[i];
  for (int i = 0; i < n; i++) {
    if (a[i] > 0 && a[i] >= a[k - 1])
      dem++;
  }
  cout << dem;

  return 0;
}
