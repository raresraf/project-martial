#include <bits/stdc++.h>

using namespace std;
int main() {
  char s[101];
  char hello[6] = "hello";
  int check = 0;
  scanf("%s", &s);
  int len = strlen(s);
  for (int i = 0; i < len; i++) {
    if (s[i] == hello[check]) {
      check += 1;
      if (check == 5)
        break;
    }
  }

  if (check == 5)
    printf("YES\n");
  else
    printf("NO\n");
}