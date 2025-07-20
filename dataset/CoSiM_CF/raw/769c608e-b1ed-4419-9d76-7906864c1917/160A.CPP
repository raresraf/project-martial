// https://codeforces.com/problemset/problem/160/A
// SOLUTION BY DN54321

#include <algorithm>
#include <ctype.h>
#include <functional>
#include <iostream>
#include <string.h>
#include <vector>

typedef long long ll;
using namespace std;

int main() {
  int n;
  scanf("%d", &n);
  int t = 0;
  int min = 0;
  int tot = 0;
  vector<int> v(n);
  for (int i = 0; i < n; ++i) {
    scanf("%d", &v[i]);
    t += v[i];
  }
  sort(v.begin(), v.end(), greater<int>());
  for (int i = 0; i < n; ++i) {
    tot += v[i];
    t -= v[i];
    ++min;
    if (tot > t)
      break;
  }
  printf("%d\n", min);

  return 0;
}