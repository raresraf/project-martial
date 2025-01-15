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
  int n, a, b;

  while (scanf("%d", &n) == 1) {
    int sum = 0, maxsum = 0;
    for (int i = 0; i < n; i++) {
      cin >> a >> b;
      sum = (sum - a) + b;

      if (sum > maxsum)
        maxsum = sum;
    }
    printf("%d\n", maxsum);
  }
  return 0;
}
