#include <algorithm>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#define LL long long
#define P(N) printf("%d\n", N);
#define S(N) scanf("%d", &N);
#define pb push_back
using namespace std;
main() {
  int n, i;
  int total_sum = 0, cur_sum = 0;
  int a[102];
  cin >> n;
  for (i = 0; i < n; i++) {
    S(a[i]);
    total_sum += a[i];
  }
  sort(a, a + n);
  for (i = n - 1; i >= 0; i--) {
    if (cur_sum > total_sum)
      break;
    cur_sum += a[i];
    total_sum -= a[i];
  }
  cout << (n - 1 - i) << endl;
  return 0;
}
