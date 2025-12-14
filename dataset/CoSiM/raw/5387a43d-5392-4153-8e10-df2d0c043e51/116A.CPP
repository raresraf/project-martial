#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, x, y, ans = 0, sum = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y;
    sum -= x;
    sum += y;
    ans = max(ans, sum);
  }
  cout << ans << endl;
}