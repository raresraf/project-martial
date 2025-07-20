#include <iostream>
using namespace std;
int main() {
  int n, cnt;
  char s, f, l, str[105];
  scanf("%d", &n);
  getchar();
  while (n--) {
    cnt = 0;
    s = getchar();
    f = s;
    while (s != '\n') {
      str[cnt] = s;
      cnt++;
      s = getchar();
    }
    str[cnt] = '\0';
    if (cnt > 10)
      printf("%c%d%c\n", str[0], cnt - 2, str[cnt - 1]);
    else
      cout << str << "\n";
  }
}