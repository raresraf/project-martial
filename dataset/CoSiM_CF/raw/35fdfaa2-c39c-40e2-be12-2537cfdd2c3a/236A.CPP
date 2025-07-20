/// Problem Name: Boy or Girl
/// Problem Link: https://codeforces.com/problemset/problem/236/A

#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

int main() {
  char s[101], ch[101];
  int test1, test2 = 0, sa = 0, n = 0;
  scanf(" %[^\n]", s);
  int len = strlen(s);
  ch[0] = 65;
  for (int i = 0; i < len; i++) {
    test1 = 0;
    for (int j = 0; j < len; j++) {
      if (i != j)
        if (s[i] == s[j])
          test1 = 1;
    }
    if (test1 == 0)
      sa++;
    else {
      int t = 0;
      n++;
      for (int k = 0; k < n; k++) {
        if (s[i] == ch[k])
          t = 1;
      }
      if (t == 0)
        test2++;
      ch[n] = s[i];
    }
  }
  if ((test2 + sa) % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");

  return 0;
}
