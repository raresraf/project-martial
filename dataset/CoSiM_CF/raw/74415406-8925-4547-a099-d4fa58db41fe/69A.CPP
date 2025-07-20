#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long

using namespace std;

int N, x, y, z;
int fx, fy, fz;

int main() {
  scanf("%d", &N);
  fx = fy = fz = 0;
  while (N--) {
    scanf("%d %d %d", &x, &y, &z);
    fx += x, fy += y, fz += z;
  }
  if (!fx && !fy && !fz)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
