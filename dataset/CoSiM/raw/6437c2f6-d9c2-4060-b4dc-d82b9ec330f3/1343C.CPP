#include <bits/stdc++.h>
#include <iostream>
using namespace std;
#define ll long long
#define sd long long

bool find_(int x, int y) {
  if ((x > 0 && y > 0) || (x < 0 && y < 0))
    return true;
  else
    return false;
}
int main() {
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    int a[n + 1];
    for (int i = 0; i < n; i++)
      cin >> a[i];

    int i = 0, j = 0;
    ll sum = 0;
    while (i < n) {
      j = i;
      int num = a[j];
      while (j + 1 < n && find_(a[j], a[j + 1])) {
        num = max(num, a[j + 1]);
        j++;
      }
      sum += num;
      i = j + 1;
    }
    cout << sum << endl;
  }
  return 0;
}
