#include <algorithm>
#include <cstdio>
using namespace std;
bool cmp(int a, int b) { return a > b; }
int main() {
  int n, a[101], i, x = 0, y = 0;
  scanf("%d", &n);
  for (i = 1; i <= n; i++) {
    scanf("%d", &a[i]);
    x += a[i];
  }
  sort(a + 1, a + n + 1, cmp);
  i = 0;
  while (y <= (x) / 2) {
    i++;
    y += a[i];
  }
  printf("%d", i);
}
