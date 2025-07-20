/// Problem Name: Beautiful Matrix
/// Problem Link: https://codeforces.com/problemset/problem/263/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int x, y, n;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      scanf("%d", &n);
      if (n) {
        x = i + 1;
        y = j + 1;
      }
    }
  }
  printf("%d\n", abs(x - 3) + abs(y - 3));
  return 0;
}
