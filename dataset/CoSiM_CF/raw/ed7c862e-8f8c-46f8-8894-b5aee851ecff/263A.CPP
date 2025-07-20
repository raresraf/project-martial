#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;

#define gc getchar_unlocked
#define ff first
#define ss second
#define ll long long
#define pb push_back
#define mp make_pair
#define pii pair<int, int>
#define vi vector<int>
#define mii map<int, int>
#define pqb priority_queue<int>
#define pqs priority_queue<int, vi, greater<int>>
#define setbits(x) __builtin_popcountll(x)
#define zrobits(x) __builtin_ctzll(x)
#define mod 1000000007
#define inf 1e18
#define ps(x, y) fixed << setprecision(y) << x
#define mk(arr, n, type) type *arr = new type[n];
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#define all(x) x.begin(), x.end()
#define clr(x) memset(x, 0, sizeof(x))
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl

typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    pbds;

void loadInputs() {
  // fast input-output
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
}

int32_t main() {

  loadInputs();

  // SOlved using the tutorial
  // if single1 is located on the intersection
  // of the r-th row and the c-th column (1-based numeration),
  // then the answer is |3-r|+|3-c|

  int x;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> x;
      if (x == 1) {
        cout << abs(i - 3) + abs(j - 3) << endl;
      }
    }
  }

  return 0;
}
