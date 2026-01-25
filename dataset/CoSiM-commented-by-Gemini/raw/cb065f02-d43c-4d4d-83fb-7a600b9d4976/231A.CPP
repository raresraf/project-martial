#include <iostream>

using namespace std;

int main() {
  int n = 0, x = 0;
  bool arr[3];

  cin >> n;

  for (int i = 0; i < n; i++) {
    for (int a = 0; a < 3; a++) {
      cin >> arr[a];
    }
    if (arr[0] && arr[2] == true) {
      x++;
    } else if (arr[1] && arr[0] == true) {
      x++;
    } else if (arr[1] && arr[2] == true) {
      x++;
    }
  }
  cout << x;
}
