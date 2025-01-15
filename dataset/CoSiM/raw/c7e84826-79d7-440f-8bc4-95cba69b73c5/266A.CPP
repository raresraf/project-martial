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
  string s;
  cin >> n >> s;
  if (n == 1) {
    printf("0\n");
    return 0;
  }

  int ans = 0;
  for (int i = 1; i < n; i++) {
    if (s[i] == s[i - 1])
      ans++;
  }
  printf("%d\n", ans);
  return 0;
}
