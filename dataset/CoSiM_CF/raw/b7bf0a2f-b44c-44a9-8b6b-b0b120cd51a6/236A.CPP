#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
#define sf scanf
#define pf printf
typedef long long ll;

int main() {
  char s[105];
  cin >> s;
  int len = strlen(s);
  sort(s, s + len);
  int count = 0;

  for (int i = 0; i < len; i++) {
    if (s[i] != s[i + 1])
      count++;
  }

  if (count % 2 == 0)
    pf("CHAT WITH HER!\n");
  else
    pf("IGNORE HIM!\n");
}
