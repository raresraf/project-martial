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

int N, x;
bool ans;

int main() {
  scanf("%d", &N);
  ans = false;
  for (x = 2; 2 * x <= N; x += 2)
    if (!((N - x) & 1))
      ans = true;
  printf("%s\n", ans ? "YES" : "NO");
  return 0;
}
