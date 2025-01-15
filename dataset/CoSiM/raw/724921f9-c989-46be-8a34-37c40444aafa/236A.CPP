#pragma GCC optimize("ofsat")
#pragma GCC target("avx,avx2,fma")

#include <bits/stdc++.h>
//#include <ext/pb_ds/assoc_container.hpp>
// using namespace __gnu_pbds;
using namespace std;

#define ui unsigned int
#define ll long long int
#define ld long double
#define ull unsigned long long int

#define ff first
#define ss second
#define pb push_back
#define mp make_pair

#define pii pair<int, int>
#define vi vector<int>
#define mii map<int, int>
#define pqb priority_queue<int>
#define pqs priority_queue<int, vi, greater<int>> // read??

#define setbits(x) __builtin_popcountll(x) // return total setbit
#define zrobits(x) __builtin_ctzll(x) // return total 0 until 1 found
#define ps(x, y) fixed << setprecision(y) << x
#define mk(arr, n, type) type *arr = new type[n];
#define sz(x) (int)x.size()
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)

#define arrsort(ar, n) sort(ar, ar + n);
#define vsort(v) sort(v.begin(), v.end())
#define vrev(v) reverse(v.begin(), v.end())
#define arrev(ar, n) reverse(ar, ar + n)

#define Fo(i, k, n) for (i = k; k < n ? i < n : i > n; k < n ? i += 1 : i -= 1)
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl
#define iter(it, a) for (auto it = a.begin(); it != a.end(); it++)
#define for0(i, n) for (ll i = 0; i < n; i++)

#define fast ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0)

#define PI acos(-1.0)
#define EPS 1e-12

const ll inf = 1000000000;
const ll mod = 1000000000 + 7;

// int dx4[] = { 0, 0, -1, +1 };                     int dy4[] = { +1, -1, 0, 0
// }; int dx8[] = { 1, 1, 0, -1, -1, -1, 0, 1, 0 };     int dy8[] = { 0, 1, 1,
// 1, 0, -1, -1, -1, 0 };

// template<typename... T>
// void read(T&... args)
// {
// 	((cin >> args), ...);
// }

// template<typename... T>
// void write(T&&... args)
// {
// 	((cout << args <<" "), ...);
// }

// mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// typedef tree<int, null_type, less<int>, rb_tree_tag,
// tree_order_statistics_node_update> pbds;

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin); // No need to comment out
  freopen("output.txt", "w", stdout);
#endif
  fast;
  //-------------------- Bismillah  ------------->

  vector<char> v;
  vector<char>::iterator it;

  string s;
  cin >> s;
  for0(i, sz(s)) v.pb(s[i]);

  sort(v.begin(), v.end()); // first need to sort the array.

  it = unique(v.begin(), v.end()); // discard all the duplicate elements.

  v.resize(distance(v.begin(), it)); // Then resize the vector.

  // iter(it,v) cout<<*it<<" ";

  if (!(sz(v) & 1))
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
