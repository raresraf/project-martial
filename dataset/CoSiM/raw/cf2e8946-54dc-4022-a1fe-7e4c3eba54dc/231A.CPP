#include <iostream>

using namespace std;

int main() {
  int n, a[5], flag = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[j];
    }
    int count = 0;
    for (int j = 0; j < 3; j++) {
      if (a[j] == 1) {
        count++;

        if (count >= 2) {
          flag++;
          break;
        }
      }
    }
  }
  cout << flag;
  return 0;
}
