#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int ev, od;
  cin >> ev >> od;
  if (ev == od || ev == od + 1 || ev + 1 == od) {
    if (ev == 0 && od == 0) {
      cout << "NO";
    } else {
      cout << "YES";
    }
  } else {
    cout << "NO";
  }
  return 0;
}
