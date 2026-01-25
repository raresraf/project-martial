#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

typedef unsigned long long ULL;
typedef long long LL;

#define REP(i, n) FOR(i, 0, n)
#define FOR(i, a, b) for (int i = a; i < b; i++)
#define ROF(i, a, b) for (int i = a; i > b; i--)
#define GI                                                                     \
  ({                                                                           \
    int t;                                                                     \
    scanf("%d", &t);                                                           \
    t;                                                                         \
  })
#define GL                                                                     \
  ({                                                                           \
    LL t;                                                                      \
    cin >> t;                                                                  \
    t;                                                                         \
  })
#define GD                                                                     \
  ({                                                                           \
    double t;                                                                  \
    scanf("%lf", &t);                                                          \
    t;                                                                         \
  })
#define pb push_back
#define mp make_pair
#define MOD 1000000007
#define INF (int)1e8
#define EPS 1e-9
#define TR(a, it)                                                              \
  for (typeof((a).begin()) it = (a).begin(); it != (a).end(); ++it)
int ans[17][17];
int solve(int m, int n) {
  int sum = m + n;
  m = max(m, n);
  n = sum - m;

  if (m < 2 || n < 1)
    return 0;
  if (ans[m][n] != -1)
    return ans[m][n];
  int t1 = 0, t2 = 0;
  if (m >= 2 && n >= 1)
    t1 = solve(m, n - 1) + solve(m - 2, 1);
  if (m >= 1 && n >= 2)
    t2 = solve(m, n - 2) + solve(m - 1, 2);

  int res = 1 + max(t1, t2);
  return res;
}
int main() {
  FOR(i, 0, 17)
  FOR(j, 0, 17)
  ans[i][j] = -1;
  int m = GI;
  int n = GI;
  cout << solve(m, n);

  return 0;
}