/// Problem Name: Tram
/// Problem Link: https://codeforces.com/problemset/problem/116/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, ex, en;
  int cap = 0;
  int cur = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    scanf("%d %d", &ex, &en);
    cur -= ex;
    cur += en;
    cap = max(cap, cur);
  }
  printf("%d\n", cap);
  return 0;
}
