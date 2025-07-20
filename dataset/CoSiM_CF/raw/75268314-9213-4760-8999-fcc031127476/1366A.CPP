#include <iostream>
using namespace std;

int main() {
  int t;
  long long a, b;
  cin >> t;
  while (t--) {
    cin >> a >> b;
    long long ans = 0;
    if (a > b)
      swap(a, b);
    ans = (a + b) / 3;
    if (a < ans)
      ans = a;
    cout << ans << endl;
  }
  return 0;
}
