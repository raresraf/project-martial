#include <bits/stdc++.h>

using namespace std;

int main() {
  int T, counter = 0;
  scanf("%d", &T);
  while (T--) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    int sum = a + b + c;
    if (sum >= 2)
      counter++;
    sum = 0;
  }
  printf("%d\n", counter);

  return 0;
}
