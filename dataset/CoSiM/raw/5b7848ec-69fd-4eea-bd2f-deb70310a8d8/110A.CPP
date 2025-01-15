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

// Outcome: Wrong answer for test case 6 --> try again

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

bool ifNearlyLucky(int n) {
  bool status = false;

  vector<int> arr;
  while (n != 0) {
    int p = n % 10;
    arr.pb(p);
    n = n / 10;
  }
  int count = 0;
  for (size_t i = 0; i < arr.size(); i++) {
    if (arr[i] == 4 || arr[i] == 7) {
      count++;
    }
  }
  if (count % 4 == 0 || count % 7 == 0) {
    status = true;
  } else {
    status = false;
  }
  return status;
}

int32_t main() {

  loadInputs();
  int n;
  cin >> n;

  if (ifNearlyLucky(n)) {
    cout << "YES"
         << "\n";
  } else {
    cout << "NO"
         << "\n";
  }
  return 0;
}
