#include <cstdio>

using namespace std;

int main() {
  char x[4];
  int n;
  scanf("%d", &n);
  int i, ans = 0;
  for (i = 0; i < n; i++) {
    scanf("%s", &x);
    if (x[1] == '+')
      ans++;
    else
      ans--;
  }
  printf("%d", ans);
  return 0;
}