/*May The Force Be With Me*/
#include <bits/stdc++.h>
#include <ctype.h>
#include <stdio.h>
#pragma GCC optimize("Ofast")
#define ll long long
#define MOD 1000000007
#define endl "\n"
#define vll vector<long long>
#define mll map<long long, long long>
#define pll pair<long long, long long>
#define all(c) c.begin(), c.end()
#define rall(c) c.rbegin(), c.rend()
#define pb push_back
#define f first
#define s second
#define inf INT_MAX
#define size_1d 10000000
#define size_2d 1000
// Snippets: delta, sieve, fastexp, dfs, bfs, dijkstra, floodfill

using namespace std;

ll a, b;

void Input() { cin >> a >> b; }

void Solve() {
  // 2 1, 1 2
  ll t = (a + b) / 3;
  cout << min(t, min(a, b)) << endl;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  ll T = 1;
  cin >> T;
  // ll t = 1;
  while (T--) {
    Input();
    // cout << "Case " << t << ": ";
    Solve();
    // t++;
  }
  return 0;
}
