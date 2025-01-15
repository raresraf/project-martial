#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, count = 0;
  int a, b, c;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    scanf("%d%d%d", &a, &b, &c);
    if ((a + b + c) >= 2) {
      count++;
    }
  }
  printf("%d", count);
  return 0;
}
