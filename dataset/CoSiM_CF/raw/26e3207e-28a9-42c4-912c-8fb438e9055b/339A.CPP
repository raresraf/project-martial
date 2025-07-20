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
  string s;
  vc v;

  while (cin >> s) {
    v.clear();
    int pls = 0;
    int len = s.size();
    for (int i = 0; i < len; i++) {
      if (s[i] == '1')
        v.push_back(1);
      else if (s[i] == '2')
        v.push_back(2);
      else if (s[i] == '3')
        v.push_back(3);
      else
        pls++;
    }

    sort(v.begin(), v.end());

    int sz = v.size();

    for (int i = 0; i < sz; i++) {
      printf("%d", v[i]);
      if (pls > 0) {
        printf("+");
        pls--;
      }
    }
    printf("\n");
  }
  return 0;
}
