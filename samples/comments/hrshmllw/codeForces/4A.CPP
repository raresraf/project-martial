#include <iostream>
using namespace std;
main() {
  int w;
  cin >> w;
  if (w % 2 == 0 && w > 2) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
