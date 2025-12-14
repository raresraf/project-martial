#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, a, count1 = 0, count2 = 0;
  scanf("%d", &t);
  for (int i = 1; i <= t; i++) {
    for (int j = 1; j <= 3; j++) {
      scanf("%d", &a);
      if (a == 1) {
        count1++;
      }
    }
    if (count1 >= 2) {
      count2++;
    }
    count1 = 0;
  }
  printf("%d\n", count2);

  return 0;
}
