#include <cstdio>

using namespace std;

int main() {
  int i, n, k, count = 0;
  scanf("%d %d", &n, &k);
  int ara[n];
  for (i = 0; i < n; i++) {
    scanf("%d", &ara[i]);
  }
  for (i = 0; i < n; i++) {
    if (ara[i] >= ara[k - 1] && ara[i] > 0) {
      count++;
    }
  }
  printf("%d", count);
  return 0;
}
