// Problem Solved: 5/22/15
// Last Edit: 9/29/16

#include <bits/stdc++.h>
using namespace std;

int main() {
  long long int n, k, w;
  scanf("%I64d %I64d %I64d", &k, &n, &w);
  k = (k * w * (w + 1)) / 2;
  if (k > n)
    printf("%I64d\n", k - n);
  else
    printf("0\n");
  return 0;
}
