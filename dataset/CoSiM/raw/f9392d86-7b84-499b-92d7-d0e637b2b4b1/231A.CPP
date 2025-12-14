#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {
  int n, p, v, t, ans = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> p >> v >> t;
    if (p + v + t >= 2)
      ans++;
  }
  cout << ans;
  return 0;
}
