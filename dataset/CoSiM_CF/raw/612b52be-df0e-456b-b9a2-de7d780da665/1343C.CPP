#include <bits/stdc++.h>
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>
typedef long long ll;
typedef unsigned long long ull;
#define pi 3.1415926535
#define mod 1000000007 // change when needed
using namespace std;
// using namespace __gnu_pbds;
template <typename T> T InvMod(T a, T b, T &x, T &y) {
  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }
  T x1, y1;
  T g = InvMod(b % a, a, x1, y1);
  x = y1 - (b / a) * x1;
  y = x1;
  return g;
}
ll fact(ll n) {
  if (n == 1)
    return 1;
  return (n % mod * fact(n - 1) % mod) % mod;
}
void sieve(vector<int> &v) {
  bool arr[1000004];
  memset(arr, true, sizeof arr);
  // memset(arr,0,sizeof arr);
  for (int i = 2; i * i <= 1000002; i++) {
    if (arr[i] == true) {
      v.push_back(i);
      for (int j = i * i; j <= 1000002; j += i) {
        arr[j] = false;
      }
    }
  }
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    ll arr[n];
    for (int i = 0; i < n; i++)
      cin >> arr[i];
    int sign;
    if (arr[0] < 0)
      sign = -1;
    else
      sign = 1;
    ll neg_min = INT_MIN, pos_max = INT_MIN;
    ll sum = 0;
    for (int i = 0; i < n; i++) {
      if (arr[i] < 0 && sign == -1) {
        neg_min = max(neg_min, arr[i]);
      } else if (arr[i] > 0 && sign == -1) {
        sum += neg_min;
        neg_min = INT_MIN;
        sign = 1;
        pos_max = arr[i];
      } else if (arr[i] < 0 && sign == 1) {
        sum += pos_max;
        pos_max = INT_MIN;
        sign = -1;
        neg_min = arr[i];
      } else {
        pos_max = max(pos_max, arr[i]);
      }
    }
    if (sign == -1) {
      sum += neg_min;
    } else {
      sum += pos_max;
    }
    cout << sum << "\n";
  }
}