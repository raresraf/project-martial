/**
`* @Samaritan
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
#define PII pair<int, int>
#define mp make_pair
#define pb push_back
#define INF 0x3f3f3f3f
#define fi first
#define se second

const int N = 3e5 + 5;

int main() {
  LL n;
  scanf("%lld", &n);
  LL l = 1, r = n - 1, h = n;
  //    LL ans = l * r * h;
  //    int step = 1;
  //    while (true) {
  //        if () {
  //            break;
  //        }
  //        if (step & 1) {
  //
  //        }
  //    }
  LL ans = 0;
  for (LL i = 2; i < n; ++i) {
    ans += i * (i + 1);
  }
  printf("%lld\n", ans);
  return 0;
}
