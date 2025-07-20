#include <iostream>

using namespace std;

int main() {

  int paradas, bajan, suben, p = 0, minimum = 0;
  cin >> paradas;

  for (int i = 0; i < paradas; i++) {

    cin >> bajan;
    p = p - bajan;

    cin >> suben;
    p = p + suben;
    if (p > minimum) {
      minimum = p;
    }
  }

  cout << minimum << endl;

  return 0;
}