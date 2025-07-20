/// Problem Name: Way Too Long Words
/// Problem Link: https://codeforces.com/problemset/problem/71/A

#include <stdio.h>
#include <string.h>

int main() {
  int t, i, len;
  char s[101];
  scanf("%d", &t);
  while (t--) {
    scanf(" %[^\n]", s);
    len = strlen(s);
    if (len > 10) {
      printf("%c%d%c\n", s[0], len - 2, s[len - 1]);
    } else {
      printf("%s\n", s);
    }
  }
  return 0;
}
