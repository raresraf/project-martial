#include <cstdio>

int n, a, b, c;
int main() {
  scanf("%d", &n);
  int answer = 0;
  for (int i = 0; i < n; ++i) {
    scanf("%d%d%d", &a, &b, &c);
    answer += (a + b + c) >= 2;
  }
  printf("%d\n", answer);
  return 0;
}