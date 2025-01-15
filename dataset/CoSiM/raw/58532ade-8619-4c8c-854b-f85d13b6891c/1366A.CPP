/// Problem Name: Shovels and Swords
/// Problem Link: https://codeforces.com/problemset/problem/1366/A

/**
 *  winners never quit
 **/

#include <bits/stdc++.h>

using namespace std;

#define pb push_back
#define mp make_pair
typedef long long Long;
void FastIO() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
}

const int N = 1e5;

int main() {
  FastIO();
  int tc, ca = 0;
  cin >> tc;
  while (tc--) {
    int a, b;
    cin >> a >> b;
    if (a > b)
      swap(a, b);
    int ans = 0;
    if (a + a <= b)
      ans = a;
    else {
      int cut = b / 2;
      ans = cut;
      a -= cut;
      b -= cut + cut;
      ans += (a + b) / 3;
    }
    cout << ans << '\n';
  }
  return 0;
}
