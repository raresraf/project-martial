#include <bits/stdc++.h>
#define ll long long
using namespace std;

int main() {
  int t;
  scanf(" %d", &t);
  while (t--) {
    int a, b;
    int add = 0;
    scanf(" %d %d", &a, &b);
    int ans = (((a << 1) - b) / 3) + a - (2 * ((a << 1) - b) / 3) -
              (2 * ((a << 1) - b) % 3 != 0);
    printf("%d\n", ans);
  }
  return 0;
}
