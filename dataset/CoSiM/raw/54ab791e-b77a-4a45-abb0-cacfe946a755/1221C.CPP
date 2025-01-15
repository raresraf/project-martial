#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
#define ff first
#define ss second
#define int long long int
#define pb push_back
#define mp make_pair
#define pii pair<int, int>
#define vi vector<int>
#define mii map<int, int>
#define maxheap priority_queue<int>
#define minheap priority_queue<int, vi, greater<int>>
#define mod 1000000007
#define inf 1e18
#define f(i, n) for (int i = 0; i < n; i++)
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)
#define ps(x, y) fixed << setprecision(y) << x
#define setbits(x) __builtin_popcountll(x)
#define zerobits(x) __builtin_ctzll(x)
#define PI 3.1415926535
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    pbds;

int32_t main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  w(t) {
    int c, m, x;
    cin >> c >> m >> x;
    int y = min(c, m);
    if (x >= y) {
      cout << y << endl;
    } else {
      int cnt = 0;
      cnt += x;
      c = c - x;
      m = m - x;
      // int e=max(c,m),g=min(c,m);

      int r;
      r = (m + c) / 3;
      if (r >= min(m, c))
        r = min(m, c);
      cnt += r;
      cout << cnt << endl;
    }
  }
}
