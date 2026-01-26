// Written By Deepesh Nimma
// Problem: 160A Codeforces
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define space " "
#define end1 "\n"
#define ar array

int n, i, a[100], sum = 0, ans = 0, cnt = 0;

int main() {
  cin >> n;

  for (i = 0; i < n; ++i) {
    cin >> a[i];
  }

  sort(a, a + n);

  for (i = 0; i < n; ++i) {
    sum += a[i];
  }

  sum = sum / 2;

  while (ans <= sum) {
    ++cnt;
    ans += a[n - cnt];
  }

  cout << cnt;
}
