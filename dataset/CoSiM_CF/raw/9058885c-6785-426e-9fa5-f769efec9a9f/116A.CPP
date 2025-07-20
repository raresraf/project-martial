#include <iostream>

using namespace std;

int main() {
  int n = 0, a = 0, b = 0, peopleIn = 0, peopleOut = 0, mAx = 0, total = 0,
      res = 0;
  // n the number of stops
  cin >> n;
  for (int i = 0; i < n; i++) {
    // a is the number of the passengers leave
    // b is the number of the passengers enter
    cin >> a;
    cin >> b;
    peopleOut = a;
    peopleIn = b;
    mAx = peopleIn - peopleOut;
    total = total + mAx;
    if (total > res) {
      res = total;
    }
  }
  cout << res;

  return 0;
}
