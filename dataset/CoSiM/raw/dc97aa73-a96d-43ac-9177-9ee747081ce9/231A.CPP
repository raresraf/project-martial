#include <iostream>

using namespace std;

int main() {
  int t;
  cin >> t;

  int x, y, z;

  int count = 0;

  while (t--) {
    cin >> x >> y >> z;

    if ((x + y + z) >= 2) {
      count++;
    }
  }

  cout << count << endl;

  return 0;
}