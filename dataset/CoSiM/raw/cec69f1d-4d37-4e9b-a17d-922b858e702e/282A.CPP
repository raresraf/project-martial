#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

int main() {
  int n;
  scanf("%d", &n);
  char s[5];
  int ans = 0;
  for (int i = 0; i < n; i++) {
    scanf("%s", s);
    if (s[1] == '+')
      ans++;
    else
      ans--;
  }
  printf("%d\n", ans);

  return 0;
}
