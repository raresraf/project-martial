#include <iostream>
using namespace std;
int main() {
  int n, x, count = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int surety = 0;
    for (int i = 0; i < 3; i++) {
      cin >> x;
      if (x == 1)
        surety++;
    }

    if (surety >= 2)
      count++;
  }

  cout << count << endl;
  return 0;
}
