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

char msk[200];
int x, y, len;
bool oke;

int main() {
  scanf("%s", msk);
  len = strlen(msk);
  x = -1;
  oke = true;
  do {
    x++;
    y = x;
    while ((x < len) && (msk[x] == msk[y]))
      y++;
    y--;
    if (y - x + 1 >= 7) {
      oke = false;
      break;
    }
    x = y;
  } while (x + 1 < len);
  printf("%s\n", oke ? "NO" : "YES");
  return 0;
}
