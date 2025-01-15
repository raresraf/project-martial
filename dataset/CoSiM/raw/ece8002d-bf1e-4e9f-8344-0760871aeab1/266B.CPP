// https://codeforces.com/problemset/problem/266/B
// SOLUTION BY DN54321

#include <algorithm>
#include <ctype.h>
#include <functional>
#include <iostream>
#include <string.h>
#include <vector>

typedef long long ll;
using namespace std;

int main() {
  int n, t;
  char g;
  scanf("%d%d", &n, &t);
  vector<bool> v(n);
  getchar();
  for (int i = 0; i < n; ++i) {
    scanf("%c", &g);
    if (g == 'B')
      v[i] = 1;
    else
      v[i] = 0;
  }
  for (int i = 0; i < t; ++i)
    for (int j = 0; j < n - 1; ++j) {
      if (v[j] && !v[j + 1]) {
        v[j] = 0;
        v[j + 1] = 1;
        ++j;
      }
    }
  for (int i = 0; i < n; ++i) {
    if (v[i])
      printf("B");
    else
      printf("G");
  }
  printf("\n");

  return 0;
}