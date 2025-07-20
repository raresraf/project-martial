#include <iostream>

using namespace std;

int main() {
  int n, x, y, z, counter = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;

    if (x + y + z >= 2)
      counter++;
  }

  cout << counter << endl;

  return 0;
}
