#include <iostream>

using namespace std;

int main() {
  int n, ans = 0, sure;
  cin >> n;
  while (n--) {
    int option;
    sure = 0;
    cin >> option;
    sure += option;
    cin >> option;
    sure += option;
    cin >> option;
    sure += option;
    if (sure > 1)
      ans++;
  }
  cout << ans;
  return 0;
}