#include <iostream>
#include <stdlib.h>
using namespace std;
int main() {
  int single;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> single;
      if (single == 1) {
        cout << abs(i - 2) + abs(j - 2);
        return 0;
      }
    }
  }
}
