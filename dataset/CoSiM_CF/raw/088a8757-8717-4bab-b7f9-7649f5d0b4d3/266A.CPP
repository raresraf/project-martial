/// Problem Name: Stones on the Table
/// Problem Link: https://codeforces.com/problemset/problem/266/A

#include <cstdio>
using namespace std;
int main() {
  int n, res = 0;
  char ch, ch1 = 65;
  scanf("%d", &n);
  getchar();
  while (n--) {
    scanf("%c", &ch);
    if (ch1 == ch)
      res++;
    ch1 = ch;
  }
  return printf("%d\n", res), 0;
}
