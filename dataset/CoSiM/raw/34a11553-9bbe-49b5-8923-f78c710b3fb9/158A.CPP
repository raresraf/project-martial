/// Problem Name: Next Round
/// Problem Link: https://codeforces.com/problemset/problem/158/A

#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int n, k;
  scanf("%d %d", &n, &k);
  int a[n + 1];
  for (int i = 1; i <= n; i++) {
    scanf("%d", &a[i]);
  }
  int ad = a[k];
  int res = 0;
  for (int i = 1; i <= n; i++) {
    if (a[i] >= ad && a[i])
      res++;
    else
      break;
  }
  printf("%d\n", res);
  return 0;
}
