/** 116A - Tram
 * http://codeforces.com/problemset/problem/116/A
 * Category: Implementation
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, in, out, actual = 0, total = 0;
  scanf("%d", &n);
  while (n--) {
    scanf("%d %d", &out, &in);
    actual += (in - out);
    total = max(total, actual);
  }
  printf("%d\n", total);
  return 0;
}