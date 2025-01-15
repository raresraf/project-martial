#include <cstdio>
#include <vector>

using namespace std;

int main() {
  int n;
  scanf("%d", &n);

  int k = 1;
  int total = k * (k + 1) / 2;
  while (total <= n) {
    ++k;
    total = k * (k + 1) / 2;
  }

  --k;
  total = k * (k + 1) / 2;

  vector<int> ans(k, 0);
  for (int i = 1; i <= k; ++i) {
    ans[i - 1] = i;
  }

  int extra;
  if (total > 1)
    extra = n % total;
  else
    extra = n - 1;

  int pos = k - 1;
  while (extra) {
    ++ans[pos];
    --pos;
    --extra;
  }

  printf("%lu\n", ans.size());

  for (size_t i = 0; i < ans.size(); ++i) {
    printf("%d ", ans[i]);
  }
  printf("\n");
  return 0;
}
