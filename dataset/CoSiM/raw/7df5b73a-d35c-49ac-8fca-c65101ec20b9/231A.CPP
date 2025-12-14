#include <cmath>
#include <iostream>
using namespace std;
int main() {
  long long int a, b, c, i, j, key = 0, cot = 0, sum;
  long long vim;
  while (!(cin >> j).eof()) {
    for (i = 1; i <= j; i++) {
      cin >> a >> b >> c;
      if (a && b == 1) {
        key++;
      } else if (b && c == 1) {
        key++;
      } else if (a && c == 1) {
        key++;
      } else if (a && b && c == 1) {
        cot++;
      } else if (a || b || c == 1) {
        sum = 0;
      }
    }
    // vim=key+cot+sum;
    cout << key + cot << endl;
    key = 0;
    cot = 0;
  }
}
