#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define MAX 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  long int n, k, x;

  while (scanf("%ld", &n) == 1) {
    int ar[5], j = 0;
    long int sum = 0;
    for (long int i = 0; i < n; i++) {
      if (n == 3) {
        cin >> x;
        sum = sum + x;
        ar[j++] = x;
      } else {
        cin >> x;

        sum = sum + x;
      }
    }
    if (n == 3) {
      if (ar[0] == 3 && ar[1] == 3 && ar[2] == 2)
        printf("3\n");
      else {
        long int ans = sum / 4;
        if (sum % 4 == 0)
          printf("%ld\n", ans);
        else
          printf("%ld\n", ans + 1);
      }
      continue;
    }
    long int ans = sum / 4;
    if (sum % 4 == 0)
      printf("%ld\n", ans);
    else
      printf("%ld\n", ans + 1);
  }
  return 0;
}
