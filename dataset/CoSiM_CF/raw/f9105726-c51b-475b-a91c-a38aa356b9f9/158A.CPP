// http://codeforces.com/problemset/problem/158/A

#include <iostream>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int n, k;
  cin >> n >> k;
  int a[n + 1];

  for (int i = 1; i <= n; i++)
    cin >> a[i];

  int res = 0;

  for (int i = 1; i <= n; i++)
    if (a[i] >= a[k] && a[i] > 0)
      res++;
  cout << res;

  return 0;
}
