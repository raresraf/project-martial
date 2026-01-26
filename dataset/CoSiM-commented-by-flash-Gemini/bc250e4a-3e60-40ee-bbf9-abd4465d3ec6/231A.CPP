// http://codeforces.com/problemset/problem/231/A
// SOLUTION BY DN54321

#include <iostream>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  int n;
  scanf("%d", &n);
  int res = 0, i, j, k;
  for (int l = 0; l < n; ++l) {
    scanf("%d%d%d", &i, &j, &k);
    if (i + j + k > 1)
      ++res;
  }
  printf("%d\n", res);
  return 0;
}