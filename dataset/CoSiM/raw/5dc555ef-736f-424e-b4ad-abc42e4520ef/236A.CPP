#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;
int n, p;
int main() {
  int i, j, cas;
  scanf("%d", &cas);
  while (cas--) {
    scanf("%d%d", &n, &p);
    int m = 2 * n + p;
    for (i = 1; i <= n && m; i++)
      for (j = i + 1; j <= n && m; j++)
        printf("%d %d\n", i, j), m--;
  }
  return 0;
}
