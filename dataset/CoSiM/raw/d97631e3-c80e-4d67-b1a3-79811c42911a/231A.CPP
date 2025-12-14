#include <iostream>

using namespace std;

int main() {

  int n;
  cin >> n;
  int p, v, t;
  int ans = 0;

  while (n--) {

    cin >> p >> v >> t;
    if (p + v + t >= 2)
      ans++;
  }

  cout << ans << endl;
}