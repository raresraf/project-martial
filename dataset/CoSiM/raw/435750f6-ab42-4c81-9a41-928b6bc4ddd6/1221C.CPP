#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pi pair<int, int>

int main() {
  int n;
  scanf("%d", &n);
  while (n--) {
    int c, m, x;
    scanf("%d %d %d", &c, &m, &x);
    int lo = 0, hi = 1e9;
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (c >= mid && m >= mid && c + m + x >= 3 * mid) {
        lo = mid;
      } else
        hi = mid - 1;
    }
    printf("%d\n", lo);
  }
}
