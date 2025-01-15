// 158A.cpp
// A. Next Round
// Author: derekhh
// May 31, 2015

#include <iostream>
using namespace std;

int a[50];

int main() {
  int n, k;
  cin >> n >> k;
  int ans = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    if (a[i] > 0 && (i < k || a[i] == a[k - 1]))
      ans++;
  }
  cout << ans << endl;
  return 0;
}