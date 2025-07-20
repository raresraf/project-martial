#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  int res = 0;
  if (a[k - 1] == 0) {
    for (int i = 0; i < k - 1; i++) {
      if (a[i] > 0)
        res++;
    }
    cout << res;
  }

  else {
    res = k;
    for (int i = k; i < n; i++) {
      if (a[i] >= a[k - 1])
        res++;
    }
    cout << res;
  }
}
