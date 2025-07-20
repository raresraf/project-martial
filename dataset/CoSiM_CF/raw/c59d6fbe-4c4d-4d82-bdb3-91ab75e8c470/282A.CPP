// https://codeforces.com/problemset/problem/282/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  int n;
  int res = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; ++i) {
    char s[100];
    scanf("%s", &s);
    if (strstr(s, "++"))
      ++res;
    else if (strstr(s, "--"))
      --res;
  }
  printf("%d\n", res);
  return 0;
}