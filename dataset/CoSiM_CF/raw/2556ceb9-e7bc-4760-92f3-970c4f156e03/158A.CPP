#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k, ans = 0, ai;
  vector<int> p;
  scanf("%d %d", &n, &k);
  for (int i = 0; i < n; i++)
    scanf("%d", &ai), p.push_back(ai);
  for (int i = 0; i < n; i++)
    if (p[i] >= p[k - 1] && p[i] > 0)
      ans++;
  printf("%d", ans);
  return 0;
}