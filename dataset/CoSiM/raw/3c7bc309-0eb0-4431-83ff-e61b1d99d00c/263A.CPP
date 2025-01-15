#include <iostream>
using namespace std;

int main(void) {
  int m, n;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      int t;
      cin >> t;
      if (t == 1) {
        m = i;
        n = j;
      }
    }
  cout << abs(2 - n) + abs(2 - m) << endl;

  return 0;
}