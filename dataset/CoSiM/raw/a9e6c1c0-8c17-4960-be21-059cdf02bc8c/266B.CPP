#include <cstdio>
#include <iostream>
using namespace std;
int main() {
  int kids, seconds;
  cin >> kids >> seconds;

  char order[kids];
  cin >> order;

  // This can be further optimised by reducing the kids checked.
  // After each iteration, we can do something about the last kid
  // and probably skip the particular index.

  for (int i = 0; i < seconds; i++) {

    for (int j = 0, k = kids - 1; j < k; j++) {
      if (order[j] == 'B' and order[j + 1] == 'G') {
        order[j + 1] = 'B';
        order[j] = 'G';
        j++;
      }
    }
  }

  cout << order;

  return 0;
}
