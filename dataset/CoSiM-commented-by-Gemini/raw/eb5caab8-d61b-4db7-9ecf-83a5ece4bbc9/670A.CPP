#include <iostream>
using namespace std;

int main() {
  int n, minD, maxD;
  cin >> n;
  // worst solution
  if (n > 7) {
    // minD = 2; maxD = 2;
    // minimum
    minD = (n / 7) * 2;
    if (n % 7 > 5)
      minD++;
    maxD = 0;
    maxD += 2;
    n -= 2;
    maxD += (n / 7) * 2;
    if (n % 7 > 5)
      maxD++;
  } else {
    if (n <= 5) {
      minD = 0;
      if (n <= 2)
        maxD = n;
      else
        maxD = 2;
    } else {
      if (n == 6)
        minD = 1;
      else
        minD = 2;
      maxD = 2;
    }
  }
  cout << minD << " " << maxD << endl;
  return 0;
}
