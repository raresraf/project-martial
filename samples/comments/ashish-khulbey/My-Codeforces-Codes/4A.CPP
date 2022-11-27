// Problem Link : https://codeforces.com/contest/4/problem/A

#include <bits/stdc++.h>
#define ll long long

using namespace std;

void solve() {
  ll n = 0;
  cin >> n;
  if (n % 2 == 0) {
    if (n < 3) {
      cout << "NO" << endl;
    } else {
      cout << "YES" << endl;
    }
  } else {
    cout << "NO" << endl;
  }
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  solve();

  return 0;
}
