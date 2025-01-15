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

char msk[30];
int cnt, len, x, y;

int main() {
  scanf("%s", msk);
  len = strlen(msk);
  cnt = 0;
  for (x = 0; x < len; x++)
    if ((msk[x] == '4') || (msk[x] == '7'))
      cnt++;

  if ((cnt == 4) || (cnt == 7))
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
