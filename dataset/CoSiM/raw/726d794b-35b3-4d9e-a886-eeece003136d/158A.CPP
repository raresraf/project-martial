/**
 * Created by LonelyEnvoy on 2017-7-23.
 * A. Next Round
 * Keywords: simple implementation
 */

#include <cstdio>
using namespace std;

#define rep(i, a, n) for (int i = a; i < n; i++)
#define erep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i++)

#define MAX 51

int main() {
  int n, k;
  int scores[MAX];
  int bound;
  int count = 0;
  scanf("%d%d", &n, &k);
  rep(i, 0, n) scanf("%d", scores + i);
  bound = scores[k - 1];

  rep(i, 0, n) {
    if (scores[i] >= bound && scores[i] > 0)
      count++;
    else
      break;
  }
  printf("%d\n", count);
}
