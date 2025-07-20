// https://codeforces.com/problemset/problem/266/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  int n;
  char s[100];
  scanf("%d", &n);
  scanf("%s", &s);
  int res = 0;
  int c = -1;
  for (int i = 0; i < n; ++i) {
    if (s[i] != c)
      c = s[i];
    else
      ++res;
  }
  printf("%d", res);
  return 0;
}