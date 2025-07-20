// http://codeforces.com/problemset/problem/158/A
// SOLUTION BY DN54321

#include <iostream>
#include <vector>
using namespace std;
int main() {
  int n, k;
  scanf("%d%d", &n, &k);
  vector<int> v(n);
  int res = 0;
  for (int i = 0; i < n; ++i) {
    scanf("%d", &v[i]);
    if (v[i] > 0 && i + 1 <= k)
      ++res;
    else if (v[i] == v[k - 1] && v[i] > 0)
      ++res;
  }
  printf("%d", res);

  return 0;
}