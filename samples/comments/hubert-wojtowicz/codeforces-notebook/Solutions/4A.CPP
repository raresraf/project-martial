// https://codeforces.com/problemset/problem/4/A
#include <bits/stdc++.h>
using namespace std;

int main() {
  int w;
  cin >> w;
  if (w < 4)
    printf("NO");
  else if (w % 2 == 0)
    printf("YES");
  else
    printf("NO");
  return 0;
}
