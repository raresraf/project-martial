// https://codeforces.com/problemset/problem/236/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  bool c[26] = {0};
  char s[200];
  int res = 0;
  scanf("%s", s);
  for (int i = 0; i < strlen(s); ++i) {
    if (!c[s[i] - 'a']) {
      c[s[i] - 'a'] = 1;
      ++res;
    }
  }
  if (res % 2)
    printf("IGNORE HIM!\n");
  else
    printf("CHAT WITH HER!\n");
  return 0;
}