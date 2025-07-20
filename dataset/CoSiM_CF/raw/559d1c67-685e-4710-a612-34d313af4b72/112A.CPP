// https://codeforces.com/problemset/problem/112/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  char s1[200], s2[200];
  scanf("%s\n%s", s1, s2);
  int sz = strlen(s1);
  for (int i = 0; i < sz; ++i) {
    if (tolower(s2[i]) > tolower(s1[i])) {
      printf("-1");
      return 0;
    } else if (tolower(s2[i]) < tolower(s1[i])) {
      printf("1");
      return 0;
    }
  }
  printf("0");
  return 0;
}