#include <algorithm>
#include <iostream>
using namespace std;

int t;
int a, b;

int main() {
  cin >> t;

  while (t--) {
    int ans = 0;

    cin >> a >> b;

    if (a > b) {
      swap(a, b);
    }

    ans = min(a, (a + b) / 3);

    cout << ans << endl;
  }
}
