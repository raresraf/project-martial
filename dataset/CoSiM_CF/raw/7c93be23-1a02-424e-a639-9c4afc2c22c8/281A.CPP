/*May The Force Be With Me*/
#include <bits/stdc++.h>
#include <ctype.h>
#include <stdio.h>
#define ll long long
#define MOD 1000000007
#define endl "\n"
#define vll vector<long long>
#define pll pair<long long, long long>
#define all(c) c.begin(), c.end()
#define f first
#define s second
#define inf INT_MAX
#define size_1d 10000000
#define size_2d 10000
// Snippets: bigint, bsearch, graph, splitstring, segtree, delta

using namespace std;

string str;

void Input() { cin >> str; }

void Solve() {
  str[0] = toupper(str[0]);
  cout << str << endl;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  ll T = 1;
  // cin >> T;
  // ll t = 1;
  while (T--) {
    Input();
    // cout << "Case " << t << ": ";
    Solve();
    // t++;
  }
  return 0;
}