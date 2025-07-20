/*
        // Author: Suraj Bora
        // Email: surajbora021@gmail.com

*/

#include <algorithm>
#include <bits/stdc++.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

// some typedefs

// typedef unsigned long long int ulli;
typedef long long int lli;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<lli> vlli;
typedef vector<char> vc;
typedef vector<pii> vpii;
typedef set<int> si;
typedef vector<string> vs;
typedef priority_queue<int> pq;
typedef priority_queue<int, std::vector<int>, std::greater<int>> mpq;

const double PI = 3.1415926535897932384626433832795028841971;

// some macros

#define PINF INT_MAX
#define NINF INT_MIN

#define FOR(i, a, b) for (int i = a; i <= (b); ++i)
#define rep(i, n) for (int i = 0; i < (n); ++i)
#define fori(i, x) for (typeof(x.begin()) i = x.begin(); i != x.end(); ++i)

#define f first
#define s second
#define pb push_back
#define mp make_pair
#define sz(x) (int)(x.size())
#define in insert
#define init(x, v) memset(x, v, sizeof x)

// freopen( "inputLA.in", "r", stdin );
// freopen( "outputLA.txt", "w", stdout );
// __gcd( 2, 4 );

int main() {

  ll n, w, k;

  cin >> k >> n >> w;

  ll total_cost = k * w * (w + 1) / 2;

  if (total_cost > n)
    cout << total_cost - n << endl;
  else
    cout << 0 << endl;
}
