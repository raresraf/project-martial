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
int x, len;

int main() {
  scanf("%s", msk);
  len = strlen(msk);
  for (x = 0; x < len; x++)
    if ((msk[x] >= 'A') && (msk[x] <= 'Z'))
      msk[x] ^= 32;
  for (x = 0; x < len; x++)
    if ((msk[x] != 'a') && (msk[x] != 'e') && (msk[x] != 'i') &&
        (msk[x] != 'u') && (msk[x] != 'o') && (msk[x] != 'y'))
      printf(".%c", msk[x]);
  printf("\n");
  return 0;
}
