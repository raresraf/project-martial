// CodeForces
// Theatre Square
// GNU C++ 17

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

int main() {
  long long n = 0, k = 0, studentsToNextRound = 0, a = 0, b = 0;
  cin >> n >> k;

  for (int i = 0; i < k; ++i) {
    cin >> a;
    if (a == 0) {
      cout << i << endl;
      return 0;
    }
  }

  for (int i = k; i < n; i++) {
    cin >> b;
    if (b != a) {
      cout << i << endl;
      return 0;
    }
  }
  cout << n << endl;
  return 0;
}