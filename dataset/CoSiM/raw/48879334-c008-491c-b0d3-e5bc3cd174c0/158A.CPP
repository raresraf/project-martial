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

int N, K, x, y, ans;
int arr[60];

int main() {
  scanf("%d %d", &N, &K);
  for (x = 1; x <= N; x++)
    scanf("%d", &arr[x]);

  ans = 0;
  for (x = 1; (x <= N) && (arr[x] >= arr[K]); x++) {
    if (arr[x] == 0)
      break;
    ans++;
  }

  printf("%d\n", ans);
  return 0;
}
