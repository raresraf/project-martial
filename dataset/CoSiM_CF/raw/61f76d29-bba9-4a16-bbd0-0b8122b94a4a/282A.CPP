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

int T;
string msk;

int main() {
  scanf("%d", &T);
  int ans = 0;

  while (T--) {
    cin >> msk;
    for (int i = 0, len = msk.length(); i < len; i++) {
      if (msk[i] == 'X')
        continue;

      if (msk[i] == '-') {
        ans--;
        i += 1;
      }

      if (msk[i] == '+') {
        ans++;
        i += 1;
      }
    }
  }

  printf("%d\n", ans);

  return 0;
}
