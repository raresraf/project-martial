#include <iostream>
using namespace std;

int main() {
  int w;
  cin >> w;
  if (w >= 1 && w <= 100 && w != 2) {

    if (w % 2 == 1) {
      cout << "NO" << endl;
    } else {
      cout << "YES" << endl;
    }

  } else {
    cout << "NO" << endl;
  }
  return 0;
}

// Attention le 2 se divise en 1 vs 1
