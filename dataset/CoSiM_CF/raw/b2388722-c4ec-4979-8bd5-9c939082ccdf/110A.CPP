#include <bits/stdc++.h>
using namespace std;
int main() {
  int count = 0;
  char s[21];
  scanf("%s", s);
  for (int i = 0; i < strlen(s); i++)
    if (s[i] == '4' || s[i] == '7')
      count++;
  switch (count) {
  case 4:
  case 7:
    printf("YES\n");
    break;
  default:
    printf("NO\n");
  }
  return 0;
}
