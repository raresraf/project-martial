#include <iostream>

using namespace std;

int main() {
  int m, n;
  cin >> m >> n;
  if (m % 2 == 0)
    cout << m * n / 2;
  else if (n % 2 == 0)
    cout << m * n / 2;
  else {
    int ans = 0;
    ans += (m / 2) * n;
    ans += n / 2;
    cout << ans;
  }
  return 0;
}