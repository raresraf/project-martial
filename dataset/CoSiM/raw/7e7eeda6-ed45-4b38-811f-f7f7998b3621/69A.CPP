/// Problem Name: Young Physicist
/// Problem Link: https://codeforces.com/problemset/problem/69/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, a, b, c, sum = 0;
  int A = 0;
  int B = 0;
  int C = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &a, &b, &c);
    A += a;
    B += b;
    C += c;
  }
  !A && !B && !C ? puts("YES") : puts("NO");
  return 0;
}
