#include <bits/stdc++.h>
using namespace std;

int main() {
  // freopen("input.txt", "r", stdin);
  char ch1[101], ch2[101];
  scanf("%s %s", ch1, ch2);
  int len = strlen(ch1);
  bool big, les;
  big = les = false;
  for (int i = 0; i < len; i++) {
    if (isupper(ch1[i]))
      ch1[i] = (int)(ch1[i] + 32);
    if (isupper(ch2[i]))
      ch2[i] = (int)(ch2[i] + 32);
    if (ch1[i] > ch2[i]) {
      big = true;
      break;
    } else if (ch1[i] < ch2[i]) {
      les = true;
      break;
    }
  }
  if (big)
    printf("1\n");
  else if (les)
    printf("-1\n");
  else
    printf("0\n");

  return 0;
}
