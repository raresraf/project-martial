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
  string s, s1;

  while (cin >> s) {
    cin >> s1;
    char ch[1005], ch1[1000];

    int j = 0, k = 0;
    int len = s.size();
    int len1 = s1.size();
    for (int i = 0; i < len; i++) {
      if (s[i] >= 'A' && s[i] <= 'Z') {
        ch[j++] = tolower(s[i]);
      } else
        ch[j++] = s[i];
    }
    for (int i = 0; i < len1; i++) {

      if (s1[i] >= 'A' && s1[i] <= 'Z') {
        ch1[k++] = tolower(s1[i]);
      } else
        ch1[k++] = s1[i];
    }
    ch[j] = '\0';
    ch1[k] = '\0';

    bool ok = true;
    int val = 0;
    for (int i = 0; i < len1; i++) {
      if (ch[i] > ch1[i]) {
        val = 1;
        break;

      } else if (ch[i] < ch1[i]) {
        val = -1;
        break;

      } else
        continue;
    }

    printf("%d\n", val);
  }
  return 0;
}
