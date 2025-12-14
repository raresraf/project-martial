#include <bits/stdc++.h>

using namespace std;

bool cmp(const int &a, const int &b) { return a > b; }

int main() {
  int n, ai, sum = 0, ansS = 0, ans = 0;
  vector<int> coin;
  scanf("%d", &n);
  for (int i = 0; i < n; i++)
    scanf("%d", &ai), coin.push_back(ai), sum += ai;
  sort(coin.begin(), coin.end(), cmp);
  for (int i = 0; i < n && ansS <= sum / 2; i++)
    ansS += coin[i], ans++;
  printf("%d", ans);
  return 0;
}