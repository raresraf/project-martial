#include <bits/stdc++.h>
using namespace std;

int main() {

  int n;
  scanf("%d", &n);
  int sum = 0, ans = 0;

  for (int i = 1; i <= n; i++) {

    int a, b;
    cin >> a >> b;
    sum += b - a;
    ans = max(ans, sum);
  }
  cout << ans << endl;

  return 0;
}
