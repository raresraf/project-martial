/**
 * Created by LonelyEnvoy on 2017-7-26.
 * A. Domino piling
 * Keywords: simple math
 */

#include <cstdio>
using namespace std;

#define rep(i, a, n) for (int i = a; i < n; i++)
#define erep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i++)

int main() {
  int m, n;
  scanf("%d%d", &m, &n);
  int result = (n % 2 == 0) ? (m * n / 2) : ((m * (n - 1)) / 2 + m / 2);
  printf("%d\n", result);
}
