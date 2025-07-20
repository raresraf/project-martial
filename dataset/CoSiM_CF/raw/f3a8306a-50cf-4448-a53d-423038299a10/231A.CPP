#include <iostream>
using namespace std;

main() {
  int n, i, total = 0;
  cin >> n;
  while (n > 0) {
    int arr[3];
    int x = 0;
    for (i = 0; i < 3; i++) {
      cin >> arr[i];
      if (arr[i] == 1)
        x = x + 1;
    }
    if (x >= 2) {
      total = total + 1;
    }
    n = n - 1;
  }
  cout << total << endl;
}
