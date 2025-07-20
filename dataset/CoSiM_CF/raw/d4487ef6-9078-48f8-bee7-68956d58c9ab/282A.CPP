#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, cnt = 0;
  char ch[1000];
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    scanf("%s", ch);
    if ((ch[0] == '-' && ch[1] == '-') || (ch[1] == '-' && ch[2] == '-'))
      cnt--;
    if ((ch[0] == '+' && ch[1] == '+') || (ch[1] == '+' && ch[2] == '+'))
      cnt++;
  }
  printf("%d", cnt);
  return 0;
}
