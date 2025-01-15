// problem : https://codeforces.com/problemset/problem/263/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int a, r, c;
  int i, j;
  for (i = 1; i <= 5; i++)
    for (j = 1; j <= 5; j++) {
      scanf("%d", &a);
      if (a) {
        r = i;
        c = j;
      }
    }

  printf("%d", abs(3 - r) + abs(3 - c));
}