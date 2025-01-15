#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;

#define LLD                                                                    \
  "%l"                                                                         \
  "ld"

#define pii pair<int, int>
#define mp make_pair<int, int>

typedef long long ll;

const int INF = 1e9;
const int N = 1e5 + 5;

// int x[N];
// int y[N];

int main() {
  int n, i, j, minA, maxA, minB, maxB, ans, x, y;

  scanf("%d", &n);
  minA = minB = INF;
  maxA = maxB = -INF;
  for (i = 0; i < n; i++) {
    scanf("%d%d", &x, &y);
    minA = min(minA, x + y - 1);
    maxA = max(maxA, x + y + 1);
    minB = min(minB, x - y - 1);
    maxB = max(maxB, x - y + 1);
  }

  ans = maxA - minA + maxB - minB;
  printf("%d\n", ans);
  return 0;
}