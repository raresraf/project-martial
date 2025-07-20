#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <streambuf>
#include <string.h>
#include <string>
#include <vector>
using namespace std;
long long sum = 0;
int num[999999];
int main() {
  long long t, x, y, n, m, count = 0, a, b;
  cin >> t;
  //    for(long long k=1;k<=t;k++)
  //    {
  sum = 0;
  long long coin = 0, val = 0;
  for (long long i = 0; i < t; i++) {
    cin >> num[i];
    sum += num[i];
  }
  sort(num, num + t);
  for (int i = t - 1; i >= 0 && val <= sum; i--) {
    coin++;
    val += num[i];
    sum -= num[i];
  }
  printf("%lld\n", coin);
  //
  //    }
  return 0;
}
