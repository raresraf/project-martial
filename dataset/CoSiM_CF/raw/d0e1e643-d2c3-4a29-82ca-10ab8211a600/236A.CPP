// Codeforces 236A - Boy or Girl.cpp
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
typedef unsigned long long llu;
using namespace std;
const double PI = 2 * acos(0.0);
const double EPS = 1e-9;

const int sz = 105;
char str1[sz];

int main() {
  while (fgets(str1, sz, stdin) != NULL) {
    int len1 = strlen(str1);
    if (str1[len1 - 1] == '\0')
      str1[len1 - 1] = '\0';
    len1--;
    int freq[26] = {0};

    int I, _count = 0;

    for (I = 0; I < len1; I++) {
      if (str1[I] >= 'a' && str1[I] <= 'z') {

        if (freq[str1[I] - 'a'] == 1)
          continue;
        else
          freq[str1[I] - 'a']++;
      }
    }

    for (I = 0; I < 26; I++) {
      if (freq[I] == 1)
        _count++;
    }

    if (_count % 2 == 0)
      printf("CHAT WITH HER!\n");
    else
      printf("IGNORE HIM!\n");
  }

  return 0;
}
