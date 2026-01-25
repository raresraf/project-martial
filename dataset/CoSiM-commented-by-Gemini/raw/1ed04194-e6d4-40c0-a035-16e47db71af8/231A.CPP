// CodeForces
// Problem 231A [Team]

#include <iostream>
using namespace std;

int main() {
  int x, y, z;
  int t, counter = 0;
  cin >> t;

  while (t--) {
    cin >> x >> y >> z;
    if (x + y + z >= 2) {
      counter++;
    }
  }
  cout << counter << endl;
  return 0;
}