// https://codeforces.com/problemset/problem/339/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  char s[200];
  int cnt[3] = {0};
  scanf("%s", &s);
  for (int i = 0; i < strlen(s); i += 2) {
    ++cnt[s[i] - '1'];
  }
  bool add = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < cnt[i]; ++j) {
      if (add)
        printf("+%d", i + 1);
      else
        printf("%d", i + 1);
      add = 1;
    }
  }
  return 0;
}