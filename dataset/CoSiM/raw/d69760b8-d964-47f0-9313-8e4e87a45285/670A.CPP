/// Problem Name: Holidays
/// Problem Link: https://codeforces.com/problemset/problem/670/A

/**
 *  winners never quit
 **/

#include <bits/stdc++.h>

using namespace std;

#define pb push_back
#define mp make_pair
typedef long long Long;

const int N = 1e6;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int tc, ca = 0;
  int n;
  cin >> n;
  int mn = n / 7 * 2;
  int mx = n / 7 * 2;
  if (n % 7) {
    if (n % 7 == 1)
      mx++;
    else
      mx += 2;
    if (n % 7 == 6)
      mn++;
  }
  cout << mn << ' ' << mx << '\n';
  return 0;
}
