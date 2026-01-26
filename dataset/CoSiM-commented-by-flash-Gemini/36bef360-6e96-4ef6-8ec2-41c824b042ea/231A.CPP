#include <iostream>
using namespace std;
int main() {
  int n, a, b, c, cnt = 0;
  scanf("%d", &n);
  while (n) {
    scanf("%d", &a);
    scanf("%d", &b);
    scanf("%d", &c);
    if (a + b + c > 1)
      cnt++;
    n--;
  }
  printf("%d", cnt);
}