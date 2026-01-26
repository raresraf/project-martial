#include <cstdio>
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int n, ans = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    if (a + b + c > 1)
      ans++;
  }
  printf("%d\n", ans);
  return 0;
}
