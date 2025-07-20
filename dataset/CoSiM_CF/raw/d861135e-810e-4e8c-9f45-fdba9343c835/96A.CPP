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

  while (cin >> s) {
    int len = s.size();
    if (len < 7) {
      printf("NO\n");
      continue;
    }
    char prev = s[0];
    int one = 0, zero = 0;
    if (prev == '1')
      one++;
    else
      zero++;
    bool ok = false;

    for (int i = 1; i < len; i++) {
      if (s[i] == '1') {
        if (prev == '1') {

          one++;
          prev = s[i];
          zero = 0;
          if (one == 7) {
            ok = true;
            break;
          }
        } else {
          one = 1;
          zero = 0;
          prev = s[i];
        }
      }

      if (s[i] == '0') {

        if (prev == '0') {
          zero++;
          prev = s[i];
          one = 0;
          if (zero == 7) {
            ok = true;
            break;
          }
        } else {
          one = 0;
          zero = 1;
          prev = s[i];
        }
      }
    }
    if (ok)
      printf("YES\n");
    else
      printf("NO\n");
  }
  return 0;
}
