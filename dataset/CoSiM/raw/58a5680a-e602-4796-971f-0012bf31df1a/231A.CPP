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
int main() {
  long long n = GL;
  int count = 0;
  FOR(i, 0, n) {
    int a = GI;
    int b = GI;
    int c = GI;
    if (a + b + c >= 2)
      count++;
  };
  cout << count;
  return 0;
}