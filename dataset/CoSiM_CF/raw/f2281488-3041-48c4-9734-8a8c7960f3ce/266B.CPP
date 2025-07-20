#include <iostream>
using namespace std;
int main() {
  int x, y;
  cin >> x >> y;
  string z;
  cin >> z;

  while (y--) {

    for (int j = 1; j < x; j++) {
      if (z[j - 1] == 'B' && z[j] == 'G') {
        z[j - 1] = 'G';
        z[j] = 'B';
        j++;
      }
    }
  }

  cout << z;
}
