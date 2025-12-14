#include <iostream>
using namespace std;
int main() {
  int n(0), ni(0), ans(0), i(0), x(0);
  cin >> n;
  for (i = 1; i <= n * 3; i++) {
    if (i % 3 == 0 && i > 0) {
      cin >> x;
      if (x == 1)
        ni++;
      if (ni >= 2) {
        ans++;
        ni = 0;
      } else
        ni = 0;
    } else {
      cin >> x;
      if (x == 1)
        ni++;
    }
  }
  cout << ans << endl;
  return 0;
}
