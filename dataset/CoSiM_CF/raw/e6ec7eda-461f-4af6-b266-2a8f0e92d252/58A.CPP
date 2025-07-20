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
  int n;
  string s;
  while (cin >> s) {

    int len = s.size();
    //        if(len<=5)
    //        {
    //            printf("NO\n");
    //            continue;
    //        }
    int h = 0, hh = 10000, e = 0, ee = 10000, l = 0, ll = 0, lll = 10000, o = 0,
        oo = 10000;
    for (int i = 0; i < len; i++) {

      if (s[i] == 'h') {
        if (i < hh) {
          hh = i;
          h = 1;
        }

      }

      else if (s[i] == 'e' && h == 1) {
        if (i < ee) {
          ee = i;
          e = 1;
        }

      }

      else if (s[i] == 'l' && h == 1 && e == 1) {
        if (i < lll) {
          lll = i;
          l = 1;
        }
        ll++;
      }

      else if (s[i] == 'o' && h == 1 && e == 1 && l == 1) {
        if (i < oo) {
          oo = i;
          o = 1;
        }
      }
    }
    // cout<<h<<e<<ll<<o<<endl;

    if (h == 1 && e == 1 && ll >= 2 && o == 1)
      printf("YES\n");
    else
      printf("NO\n");
  }
  return 0;
}
