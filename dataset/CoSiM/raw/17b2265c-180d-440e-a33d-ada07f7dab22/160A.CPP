#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n, sum = 0, count = 0, ans = 0;
  cin >> n;

  int a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }

  sum = sum / 2;
  sort(a, a + n);

  while (ans <= sum) {
    ++count;
    ans += a[n - count];
  }
  cout << count;
  return 0;
}