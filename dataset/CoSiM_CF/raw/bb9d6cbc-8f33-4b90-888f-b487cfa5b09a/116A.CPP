
// https://codeforces.com/problemset/problem/116/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>

typedef long long ll;
using namespace std;

int main() {
  int n, x, y;
  int mx = 0;
  int c = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; ++i) {
    scanf("%d%d", &x, &y);
    c += y - x;
    if (c > mx)
      mx = c;
  }
  printf("%d\n", mx);

  return 0;
}