// by mo1ein
#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, c = 0;
  cin >> n >> k;
  int a[n + 1];
  for (int i = 1; i <= n; i++)
    cin >> a[i];
  for (int i = 1; i <= n; i++)
    if (a[i] >= a[k] && a[i])
      c++;
  cout << c;
}
