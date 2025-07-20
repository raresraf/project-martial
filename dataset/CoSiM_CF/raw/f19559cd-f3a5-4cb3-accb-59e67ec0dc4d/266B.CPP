/**
 * Created by LonelyEnvoy on 2017-8-18.
 * A. Queue at the School
 * Keywords: simple strings
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;

#define rep(i, a, n) for (int i = a; i < n; ++i)
#define erep(i, a, n) for (int i = a; i <= n; ++i)
#define per(i, a, n) for (int i = n - 1; i >= a; --i)

#define MAX 51

char buf[MAX];
int q[MAX];
int n, t;

int main() {
  scanf("%d%d%s", &n, &t, buf);
  rep(i, 0, n) { q[i] = (buf[i] == 'B' ? 1 : 0); }

  rep(k, 0, t) {
    erep(i, 0, n - 2) {
      if (q[i] && !q[i + 1]) {
        q[i] = 0;
        q[i + 1] = 1;
        i++;
      }
    }
  }

  rep(i, 0, n) { printf(q[i] ? "B" : "G"); }
  printf("\n");
}
