#include <iostream>
using namespace std;
int main() {
  int n, a, b, ma = 0, cur = 0;
  scanf("%d", &n);
  for (register int i = 0; i < n; i++) {
    scanf("%d", &a);
    scanf("%d", &b);
    cur = cur - a;
    cur = cur + b;
    ma = max(cur, ma);
  }
  printf("%d", ma);
}