#include <algorithm>
#include <cstdio>
int main() {
  int t, num[101];
  while (scanf("%d", &t) == 1) {
    int sum = 0, total = 0, Ans = 0;
    for (int i = 0; i < t; i++) {
      scanf("%d", &num[i]);
      sum += num[i];
    }
    std::sort(num, num + t);
    for (int i = t - 1; i >= 0 && total <= sum / 2; i--) {
      total += num[i];
      Ans++;
    }
    printf("%d\n", Ans);
  }
  return 0;
}