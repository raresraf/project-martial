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
#define INF 1000000000

using namespace std;

char msk[111];
string a, b;
int x;

int main() {
  scanf("%s", msk);
  a = msk;
  scanf("%s", msk);
  b = msk;

  for (x = 0; x < a.size(); x++)
    if ((a[x] >= 'A') && (a[x] <= 'Z'))
      a[x] ^= 32;
  for (x = 0; x < b.size(); x++)
    if ((b[x] >= 'A') && (b[x] <= 'Z'))
      b[x] ^= 32;

  if (a == b)
    printf("0\n");
  else if (a > b)
    printf("1\n");
  else
    printf("-1\n");
  return 0;
}
