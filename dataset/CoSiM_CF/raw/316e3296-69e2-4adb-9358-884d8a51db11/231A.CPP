#include <iostream>

using namespace std;

int main() {

  int n, a;
  int k, count = 0;

  cin >> n;

  for (int i = 0; i < n; ++i) {
    k = 0;

    for (int j = 0; j < 3; ++j) {
      cin >> a;
      if (a == 1) {
        k++;
      }
    }
    if (k >= 2) {
      count++;
    }
  }

  cout << count << endl;

  return 0;
}